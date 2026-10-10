"""
nec_random112.py -- size-matched random baseline for the VLG-CBM NEC evaluation (quick fix 1).

WHY: nec_sparse_head.py compares our trained 112-concept bottleneck with a random CBL of 512 neurons
(VLG-CBM's size for CUB). At NEC = 5 the sparse layer then chooses its 5 weights per class from 512 random
directions versus only 112 concept directions, so "our concepts do not beat random" mixes two things:
concept quality and the number of candidates. This script removes the second one: the random CBL gets
exactly 112 neurons, the same as the real bottleneck. Everything else is identical to nec_sparse_head.py
(same representation model.features, W ~ N(0, 1), same seed formula, solver, path, pruning rule, data split).

STANDALONE: imports nec_sparse_head.py and sets its module-level RANDOM_CBL_NEURONS / output paths in memory
only; no file is edited. Reads the finished real-CBL units from nec_sparse/units/ (read-only).
Writes only into nec_random112/ (guarded) and checks every other repo file is unchanged.

USAGE
  python nec_random112.py             # 3 models x seeds 0-2, random-112 CBLs, all NEC targets, + report
  python nec_random112.py --report-only
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import nec_sparse_head as nsh

N_RANDOM = 112
TARGETS = (5,)                                               # the go/no-go number (ANEC-5); NEC 10-30 not needed
REAL_UNITS = nsh.UNITS_DIR                                   # nec_sparse/units (read-only)
OUT_ROOT = os.path.join(ROOT, "nec_random112")
nsh.RANDOM_CBL_NEURONS = N_RANDOM
nsh.OUT_ROOT = OUT_ROOT
nsh.UNITS_DIR = os.path.join(OUT_ROOT, "units")
nsh.LOG_PATH = os.path.join(OUT_ROOT, "nec_random112_log.txt")
MD_PATH = os.path.join(OUT_ROOT, "nec_random112_report.md")
JSON_PATH = os.path.join(OUT_ROOT, "nec_random112_report.json")
LABEL = {"learned_decoder": "SNN + GRU (ours)", "ann34_mlp": "ResNet-34 + MLP", "ann18_mlp": "ResNet-18 + MLP"}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_report(models, seeds):
    J, md = {"n_random": N_RANDOM, "models": {}}, []
    md += [f"# NEC with a size-matched random baseline ({N_RANDOM} random neurons)", "",
           "Protocol identical to `nec_sparse/nec_sparse_report.md` (VLG-CBM sparse layer, strict NEC, FISTA, "
           "alpha 0.99); only the random CBL size changes from 512 to 112, the size of our concept layer. "
           "Mean +- std over seeds 0-2; gap = trained concepts minus random-112 at NEC = 5, pooled paired "
           "bootstrap over test images (10,000 resamples).", "",
           "| Model | Trained 112 concepts ANEC-5 | Random-112 ANEC-5 | Random-512 ANEC-5 | "
           "Gap vs random-112 (95% CI) |",
           "|---|---|---|---|---|"]
    f = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
    for m in models:
        real = [load(os.path.join(REAL_UNITS, f"{m}_seed{s}.json")) for s in seeds]
        r512 = [load(os.path.join(REAL_UNITS, f"{m}_seed{s}_random.json")) for s in seeds]
        r112 = [load(nsh.unit_path(m, s, "random")) for s in seeds]
        if any(u is None for u in real + r112):
            md.append(f"| {LABEL[m]} | missing units | | | |")
            continue
        a5 = lambda us: [u["targets"]["5"]["test_acc"] for u in us]
        ca = [u["test_correct"]["5"] for u in real]
        cb = [u["test_correct"]["5"] for u in r112]
        g, lo, hi, lo90, hi90 = nsh.paired_bootstrap_pooled(ca, cb)
        J["models"][m] = {"anec5_real": a5(real), "anec5_random112": a5(r112),
                          "anec5_random512": a5(r512) if all(r512) else None,
                          "gap_anec5_real_minus_random112": {"gap": g, "ci95": [lo, hi], "ci90": [lo90, hi90]},
                          "held_out_anec5_random112": [u["targets"]["5"]["held_out_acc"] for u in r112]}
        md.append(f"| {LABEL[m]} | {f(a5(real))} | {f(a5(r112))} | "
                  f"{f(a5(r512)) if all(r512) else '-'} | {g:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    md += ["", "Reading: a CI entirely above 0 means the trained concepts beat a random layer of the same size "
           "at NEC = 5; a CI containing 0 means no detectable difference; entirely below 0 means random wins."]
    with open(nsh._safe_path(MD_PATH), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    nsh._write_json(J, JSON_PATH)
    print("\n".join(md))


def _fast_path_and_dtype():
    """Two speed-ups for a CPU run, both matching how the original units were made or exact for the headline rule:
    * float32 with tolerance 1e-5 -- what nec_sparse_head.py itself uses on CUDA, where the published real and
      random-512 units were computed (CPU would otherwise default to float64, ~3x slower);
    * only NEC = 5 (ANEC-5, the go/no-go number) is evaluated, and the path stops at the first point with NEC > 6. The reported rule (first point with NEC >= k,
      pruned to k) only looks at earlier points, so it is unchanged; the later points were never used."""
    import functools, torch
    stop_at = max(TARGETS) + 1

    def regularization_path(X, Y1h, n_lambda=nsh.N_LAMBDA, ratio=nsh.LAMBDA_RATIO, alpha=nsh.ALPHA,
                            max_iter=nsh.MAX_ITER, tol=nsh.TOL, log=None):
        C = Y1h.shape[1]
        lmax = nsh.lambda_max(X, Y1h, alpha)
        lams = np.exp(np.linspace(np.log(lmax), np.log(lmax / ratio), n_lambda))
        L = nsh.lipschitz(X)
        W = torch.zeros(C, X.shape[1], dtype=X.dtype, device=X.device)
        b = nsh.bias_only(Y1h).to(X.dtype)
        path = []
        for i, lam in enumerate(lams):
            W, b, its = nsh.fista(X, Y1h, float(lam), W, b, L, alpha, max_iter, tol)
            nec = float((W != 0).sum()) / C
            path.append((float(lam), W.clone(), b.clone(), nec, its))
            if log and i % 5 == 0:
                log(f"      path {i + 1:2d}/{n_lambda}  lambda={lam:.3e}  NEC={nec:7.2f}  iters={its}")
            if nec > stop_at:
                if log:
                    log(f"      path stopped at {i + 1}/{n_lambda} (NEC {nec:.2f} > {stop_at})")
                break
        return path
    nsh.regularization_path = regularization_path
    nsh.nec_evaluate = functools.partial(nsh.nec_evaluate, dtype=torch.float32)


CPU_TOL = {"max_abs_cs_diff": 5e-3, "pred_mismatches": 5, "acc_diff": 0.1, "auc_diff": 1e-3, "ece_diff": 1e-3}
SANITY_LOG = []


def _cpu_sanity(cam, rav):
    """The saved outputs were made on a CUDA GPU; on a CPU float32 kernels differ at ~1e-3 in a few scores.
    Same checks as calibration_all_models.sanity, with the CPU tolerances above; every result is reported."""
    def sanity(kind, seed, rule, cs, pred, cids, attrs):
        rd = rav.run_dir(kind, seed)
        z = np.load(os.path.join(rd, f"test_outputs_{rule}.npz"))
        with open(os.path.join(rd, "result.json"), encoding="utf-8") as f:
            r = json.load(f)[rule]
        acc = float((pred == cids).mean() * 100.0)
        auc, ece = cam._mean_auc(cs, attrs), cam._mean_ece(cs, attrs)
        s = {"max_abs_cs_diff": float(np.abs(cs - z["cs"]).max()), "pred_mismatches": int((pred != z["pred"]).sum()),
             "acc": acc, "acc_diff": abs(acc - r["test_acc"]), "auc": auc, "auc_diff": abs(auc - r["test_concept_auc"]),
             "ece": ece, "ece_diff": abs(ece - r["test_concept_ece"])}
        s["ok"] = all(s[k] <= v for k, v in CPU_TOL.items())
        s["tolerance"] = "cpu: " + json.dumps(CPU_TOL)
        SANITY_LOG.append({"unit": f"{kind}_seed{seed}", **s})
        if not s["ok"]:
            raise RuntimeError(f"[Sanity] {kind} seed {seed} does not reproduce the saved outputs within CPU "
                               f"tolerance: {s}")
        return s
    return sanity


def main(args):
    import calibration_all_models as cam
    import run_aug_views as rav
    import run_seeds as rs
    cam.sanity = _cpu_sanity(cam, rav)
    _fast_path_and_dtype()
    sys.stdout = nsh._Tee(nsh.LOG_PATH)
    print("=" * 72)
    print(f"  NEC RANDOM-{N_RANDOM} CONTROL  [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 72)
    before = nsh.fingerprint_protected()
    models, seeds = nsh.MAIN, rav.SEEDS
    if not args.report_only:
        feats = cam.Feats()
        n_concepts = rav.n_concepts_of(feats.rows["train_fit"])
        for m in models:
            for s in seeds:
                if load(nsh.unit_path(m, s, "random")) is not None:
                    print(f"   [{m} seed {s}] done already -- skipped")
                    continue
                nsh.run_unit(m, s, "random", feats, n_concepts, targets=TARGETS)
    if SANITY_LOG:
        nsh._write_json(SANITY_LOG, os.path.join(OUT_ROOT, "sanity_cpu.json"))
    build_report(models, seeds)
    changed = rs.compare_fingerprints(before, nsh.fingerprint_protected())
    print(f"\n[Guard] {'WARNING: changed outside nec_random112/: ' + str(changed[:10]) if changed else 'PROTECTED FILES UNCHANGED: all ' + str(len(before)) + ' pre-existing files verified.'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report-only", action="store_true")
    warnings.filterwarnings("ignore")
    main(p.parse_args())
