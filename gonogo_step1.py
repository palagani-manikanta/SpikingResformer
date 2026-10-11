"""
gonogo_step1.py -- CUB go/no-go, step 1: can a better CONCEPT LOSS make the learned concepts beat a random layer
of the same size at NEC = 5?

WHY (nec_oracle/report.md): perfect class-level concepts give 100% at NEC = 5 and even 10% random label flips give
56.3%, above random-112 (43.9-47.8%). Our trained concepts get 12.2% (SNN) / 9.8% (ResNet-34) of the 112 binary
decisions wrong. In the joint loss (lambda_concept = lambda_task = 1) the class cross-entropy is ~10x the mean
concept BCE, so the concept layer is shaped mostly by the class loss. Step 1 tests three cheap remedies.

VARIANTS (both models: SNN + GRU 'learned_decoder' and ResNet-34 + MLP 'ann34_mlp'; seeds 0, 1, 2):
  lc2, lc3 -- step 1b, added after the first run (lc5 passed (a) with a 4-point margin but failed (b)):
            joint loss with lambda_concept = 2 / 3. All variants are reported, including the failed ones.
  base   -- the published runs (aug_views/runs, joint, lambda_concept 1), read-only: not retrained.
  lc5    -- joint loss, lambda_concept = 5
  lc20   -- joint loss, lambda_concept = 20
  lc5pw  -- lambda_concept = 5 + rare-concept weighting: BCE of a positive label weighted by
            min(10, n_neg / n_pos) of that concept (train_fit counts)
  seq    -- sequential CBM (Koh et al. 2020): stage 1 trains decoder + concept layer with the concept BCE only
            (selected on held-out concept AUC); stage 2 freezes them and trains a freshly initialised class head on
            the predicted concepts (30 epochs, same optimiser, concept dropout 0.25; selected on held-out accuracy).
Everything else is run_aug_views.train_one unchanged: 8 cached augmented views, AdamW 1e-3 / 1e-4, batch 32,
50 epochs, cosine LR, grad clip 5, concept dropout 0.25, selection on the 899 held-out images.

GO/NO-GO RULE (fixed before running; checked per variant, epochs selected on held-out class accuracy):
  (a) SNN concepts beat random-112 at NEC = 5: paired-bootstrap 95% CI of (trained - random-112) above 0, pooled
      over the 3 seeds. Random-112 = 112 random N(0,1) directions on THIS variant's model.features, so the random
      baseline sees exactly the representation the trained concepts are read from.
  (b) SNN accuracy within 1 point of ResNet-34 + MLP with the same variant: 95% CI of (SNN - ANN) above -1.
  PASS = (a) and (b).

NEC protocol: nec_sparse_head.py (VLG-CBM sparse layer, elastic net alpha 0.99, FISTA, strict NEC = 5), with the
speed-ups validated in nec_random112_check.py (float32, path stopped once NEC > 6; reproduces the published units).

STANDALONE: imports run_aug_views, run_seeds_round3, calibration_all_models and nec_sparse_head read-only and
redirects run_aug_views' output root IN MEMORY to gonogo_step1/. Writes nothing else; fingerprint-checked.

USAGE (repo root, project venv python, GPU):
  python gonogo_step1.py --dry-run     # 1 epoch, 5 batches, 1 seed, NEC on one unit; writes only gonogo_step1_dry/
  python gonogo_step1.py               # full run, resumable (re-run the same command after a stop)
  python gonogo_step1.py --report      # rebuild the report from finished units
"""
import argparse, json, os, sys, time, types, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

import run_aug_views as rav
import run_seeds as rs
import run_seeds_round3 as r3
import calibration_all_models as cam
import nec_sparse_head as nsh
from train_cbm import DEVICE, cosine_lr_schedule

MODELS = ("learned_decoder", "ann34_mlp")
LABEL = {"learned_decoder": "SNN + GRU", "ann34_mlp": "ResNet-34 + MLP"}
SEEDS = (0, 1, 2)
VARIANTS = {"lc2": {"lc": 2.0, "pw": False, "seq": False},          # step 1b (added after step 1: weight 5 passed
            "lc3": {"lc": 3.0, "pw": False, "seq": False},          #   (a) with margin but failed (b))
            "lc5": {"lc": 5.0, "pw": False, "seq": False},
            "lc20": {"lc": 20.0, "pw": False, "seq": False},
            "lc5pw": {"lc": 5.0, "pw": True, "seq": False},
            "seq": {"lc": 1.0, "pw": False, "seq": True}}
STAGE2_EPOCHS = 30
PW_MAX = 10.0
N_RANDOM = 112
NEC_K = 5
BOOT_SEED, N_BOOT = 20260826, 10_000
RANDOM_SEED_BASE = nsh.RANDOM_SEED_BASE

OUT_ROOT = os.path.join(ROOT, "gonogo_step1")
REPORT_TAG = ""
_ORIG = {"OUT_ROOT": rav.OUT_ROOT, "RUNS_DIR": rav.RUNS_DIR, "make_model": r3.make_model, "cfg": rav._run_config}
CUR = {"variant": None, "lt": 1.0, "pw": None}


# ------------------------------------------------------------------------------------------------ redirection
def set_out(root):
    global OUT_ROOT
    OUT_ROOT = root
    rav.OUT_ROOT = root


def use_variant(v):
    CUR["variant"] = v
    rav.RUNS_DIR = os.path.join(OUT_ROOT, "runs", v)


def _loss(self, cs, cl, at, yt):
    """lambda_concept * BCE (optionally rare-concept weighted) + lambda_task * CE."""
    cfg = VARIANTS[CUR["variant"]]
    t = at.float()
    if cfg["pw"]:
        w = 1.0 + (CUR["pw"].to(t.device) - 1.0) * t
        bce = F.binary_cross_entropy(cs, t, weight=w, reduction="mean")
    else:
        bce = F.binary_cross_entropy(cs, t, reduction="mean")
    ce = F.cross_entropy(cl, yt.long(), reduction="mean")
    return cfg["lc"] * bce + CUR["lt"] * ce, bce, ce


def make_model(kind, n_concepts):
    m = _ORIG["make_model"](kind, n_concepts)
    m.compute_loss = types.MethodType(_loss, m)
    return m


def run_config(kind, seed, k_views=rav.K_VIEWS):
    c = _ORIG["cfg"](kind, seed, k_views)
    c["gonogo_variant"] = CUR["variant"]
    c["gonogo_settings"] = dict(VARIANTS[CUR["variant"]], lambda_task=CUR["lt"])
    return c


r3.make_model = make_model
rav._run_config = run_config


# ------------------------------------------------------------------------------------------------ training
def stage2_head(model, store, train_lab, held, seed, epochs=STAGE2_EPOCHS, max_batches=None):
    """Sequential CBM, stage 2: decoder + concept layer frozen; a fresh head trained on predicted concepts."""
    import random
    torch.manual_seed(seed + 77); np.random.seed(seed + 77); random.seed(seed + 77)
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.head
    for mod in head.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=rav.LR, weight_decay=rav.WD)
    gen = torch.Generator().manual_seed(seed + 77)
    vgen = torch.Generator().manual_seed(seed + 78)
    at = torch.from_numpy(train_lab[0]).float().to(DEVICE)
    yt = torch.from_numpy(train_lab[1]).long().to(DEVICE)
    val_batches = rs._batches(*rav._dev(*held))
    best = {"val_class_acc": -1.0}
    n = store.n
    for epoch in range(1, epochs + 1):
        cosine_lr_schedule(opt, epoch - 1, epochs, lr_max=rav.LR)
        v = torch.randint(store.k, (n,), generator=vgen).numpy()
        xt = torch.from_numpy(store.gather(v)).to(DEVICE).float()
        with torch.no_grad():
            model.eval()
            cs_all = torch.cat([model.cbl(model.features(xt[i:i + 512])) for i in range(0, n, 512)])
        head.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        nb = n // rav.BATCH_SIZE if max_batches is None else min(max_batches, n // rav.BATCH_SIZE)
        for bi in range(nb):
            idx = perm[bi * rav.BATCH_SIZE:(bi + 1) * rav.BATCH_SIZE]
            c, a = cs_all[idx], at[idx]
            mask = torch.rand_like(c) < rav.CONCEPT_DROPOUT
            loss = F.cross_entropy(head(torch.where(mask, a, c)), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        del xt, cs_all
        model.eval()
        val_auc, val_acc, _ = rav.evaluate(model, val_batches, DEVICE)
        if val_acc > best["val_class_acc"]:
            best = rav._snap(epoch, val_acc, val_auc, model)
        print(f"    stage 2 seed {seed} ep {epoch:2d}/{epochs} | held-out acc {val_acc:.2f}% (best {best['val_class_acc']:.2f}%)")
    return best


def train_variant(v, kind, seed, ev, train_lab, n_concepts, epochs, max_batches):
    use_variant(v)
    if rav.is_done(kind, seed):
        print(f"[{v}] {kind} seed {seed}: done earlier, skipped")
        return
    cfg = VARIANTS[v]
    CUR["lt"] = 0.0 if cfg["seq"] else 1.0
    print(f"\n[{v}] {LABEL[kind]} seed {seed}" + (" -- stage 1 (concepts only)" if cfg["seq"] else ""))
    store = rav.ViewStore(rav.FEATURE_KEY[kind])
    model, best, hist, _, _ = rav.train_one(kind, seed, store, train_lab, ev["held_out"], n_concepts,
                                            epochs=epochs, max_batches=max_batches)
    if cfg["seq"]:
        model.load_state_dict(best["auc"]["model_state"])           # stage 1 selected on held-out concept AUC
        b2 = stage2_head(model, store, train_lab, ev["held_out"], seed,
                         epochs=STAGE2_EPOCHS if max_batches is None else 1, max_batches=max_batches)
        best = {"acc": b2, "auc": best["auc"]}
        CUR["lt"] = 1.0
        rav._atomic_save({"config": rav._run_config(kind, seed), "rule": "acc", **b2},
                         os.path.join(rav.run_dir(kind, seed), "best_acc.pth"))      # the stage-2 model
    rav.finish_run(kind, seed, model, best, hist, ev["test"])


# ------------------------------------------------------------------------------------------------ NEC = 5
def unit_path(v, kind, seed):
    return os.path.join(OUT_ROOT, "nec", f"{v}_{kind}_seed{seed}.json")


@torch.no_grad()
def reps(model, x, bs=256):
    X = torch.from_numpy(np.ascontiguousarray(x)).float().to(DEVICE)
    H, Z = [], []
    for i in range(0, len(X), bs):
        h = model.features(X[i:i + bs])
        H.append(h.cpu().numpy()); Z.append(model.cbl.linear(h).cpu().numpy())
    return np.concatenate(H), np.concatenate(Z)


def nec_unit(v, kind, seed, feats, n_concepts):
    p = unit_path(v, kind, seed)
    if os.path.isfile(p):
        return json.load(open(p, encoding="utf-8"))
    rd = os.path.join(OUT_ROOT, "runs", v, f"{kind}_seed{seed}") if v != "base" else \
        os.path.join(_ORIG["RUNS_DIR"], f"{kind}_seed{seed}")
    ck = torch.load(os.path.join(rd, "best_acc.pth"), map_location="cpu", weights_only=False)
    model = _ORIG["make_model"](kind, n_concepts).to(DEVICE)
    model.load_state_dict(ck["model_state"]); model.eval()
    splits = ("train_fit", "held_out", "test")
    H, Z, y = {}, {}, {}
    for s in splits:
        x, a, c = feats.get(rav.FEATURE_KEY[kind], s)
        H[s], Z[s] = reps(model, x)
        y[s] = np.asarray(c).astype(np.int64)
    g = torch.Generator().manual_seed(RANDOM_SEED_BASE + 1000 * seed + sum(map(ord, kind)))
    Wr = torch.randn(N_RANDOM, H["train_fit"].shape[1], generator=g, dtype=torch.float64).numpy()
    out = {"variant": v, "model": kind, "seed": seed, "epoch": ck["epoch"]}
    for name, X in (("trained", Z), ("random112", {s: H[s].astype(np.float64) @ Wr.T for s in splits})):
        r = nsh.nec_evaluate(X["train_fit"], y["train_fit"], {"held_out": (X["held_out"], y["held_out"]),
                                                              "test": (X["test"], y["test"])},
                             200, targets=(NEC_K,), device=DEVICE, dtype=torch.float32)
        k = r["targets"][NEC_K]
        out[name] = {"test_acc": k["test_acc"], "held_out_acc": k["held_out_acc"],
                     "test_correct": (r["preds"][NEC_K]["test"] == y["test"]).tolist()}
        print(f"   NEC=5 [{v}] {LABEL[kind]} seed {seed} {name}: test {k['test_acc']:.2f}%")
    os.makedirs(os.path.dirname(rav._safe_path(p)), exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"))
    return out


def _fast_nec_path():
    """Same as nec_random112._fast_path_and_dtype: stop the path once NEC > 6 (only NEC = 5 is reported)."""
    def path(X, Y1h, n_lambda=nsh.N_LAMBDA, ratio=nsh.LAMBDA_RATIO, alpha=nsh.ALPHA, max_iter=nsh.MAX_ITER,
             tol=nsh.TOL, log=None):
        C = Y1h.shape[1]
        lmax = nsh.lambda_max(X, Y1h, alpha)
        lams = np.exp(np.linspace(np.log(lmax), np.log(lmax / ratio), n_lambda))
        L = nsh.lipschitz(X)
        W = torch.zeros(C, X.shape[1], dtype=X.dtype, device=X.device)
        b = nsh.bias_only(Y1h).to(X.dtype)
        res = []
        for lam in lams:
            W, b, its = nsh.fista(X, Y1h, float(lam), W, b, L, alpha, max_iter, tol)
            nec = float((W != 0).sum()) / C
            res.append((float(lam), W.clone(), b.clone(), nec, its))
            if nec > NEC_K + 1:
                break
        return res
    nsh.regularization_path = path


# ------------------------------------------------------------------------------------------------ report
def boot(ca, cb):
    a = np.asarray(ca, float); d = (a - np.asarray(cb, float)).mean(0) * 100
    rng = np.random.default_rng(BOOT_SEED); n = len(d)
    m = np.concatenate([d[rng.integers(0, n, size=(1000, n))].mean(1) for _ in range(N_BOOT // 1000)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_correct(v, kind, seed, cids):
    rd = os.path.join(OUT_ROOT, "runs", v, f"{kind}_seed{seed}") if v != "base" else \
        os.path.join(_ORIG["RUNS_DIR"], f"{kind}_seed{seed}")
    z = np.load(os.path.join(rd, "test_outputs_acc.npz"))
    return z["pred"] == cids, z["cs"]


def build_report(variants, seeds, feats):
    _, attrs, cids = feats.get("snn", "test")
    attrs = np.asarray(attrs); cids = np.asarray(cids)
    J, rows = {"rule": "PASS = (a) trained - random112 at NEC=5, 95% CI > 0 (SNN) and (b) SNN - ANN accuracy, "
                       "95% CI > -1 point; same variant, 3 seeds, held-out-accuracy selection", "variants": {}}, []
    for v in variants:
        U = {k: [json.load(open(unit_path(v, k, s), encoding="utf-8")) for s in seeds] for k in MODELS}
        R = {k: [run_correct(v, k, s, cids) for s in seeds] for k in MODELS}
        rec = {}
        for k in MODELS:
            acc = [c.mean() * 100 for c, _ in R[k]]
            err = [float(((cs > 0.5) != attrs).mean() * 100) for _, cs in R[k]]
            gr = boot([u["trained"]["test_correct"] for u in U[k]], [u["random112"]["test_correct"] for u in U[k]])
            rec[k] = {"acc": acc, "concept_err_pct": err,
                      "nec5_trained": [u["trained"]["test_acc"] for u in U[k]],
                      "nec5_random112": [u["random112"]["test_acc"] for u in U[k]], "gap_vs_random": gr}
        ga = boot([c for c, _ in R["learned_decoder"]], [c for c, _ in R["ann34_mlp"]])
        passed_a = rec["learned_decoder"]["gap_vs_random"][1] > 0
        passed_b = ga[1] > -1.0
        J["variants"][v] = {"models": rec, "snn_minus_ann_acc": ga, "pass_a": passed_a, "pass_b": passed_b,
                            "PASS": passed_a and passed_b}
        s, a = rec["learned_decoder"], rec["ann34_mlp"]
        f = lambda x: f"{np.mean(x):.2f}"
        rows.append(f"| {v} | {f(s['acc'])} | {f(a['acc'])} | {ga[0]:+.2f} [{ga[1]:+.2f}, {ga[2]:+.2f}] | "
                    f"{f(s['concept_err_pct'])} / {f(a['concept_err_pct'])} | {f(s['nec5_trained'])} vs {f(s['nec5_random112'])} | "
                    f"{s['gap_vs_random'][0]:+.2f} [{s['gap_vs_random'][1]:+.2f}, {s['gap_vs_random'][2]:+.2f}] | "
                    f"{f(a['nec5_trained'])} vs {f(a['nec5_random112'])} | "
                    f"{'**PASS**' if passed_a and passed_b else 'fail'} ({'a' if passed_a else '-'}{'b' if passed_b else '-'}) |")
    md = ["# Go/no-go step 1: concept loss", "",
          f"{len(seeds)} seeds, epochs selected on held-out class accuracy. NEC = 5 with the VLG-CBM sparse layer; random-112 = "
          "112 random directions on the same variant's features. Gaps: paired bootstrap over test images "
          "(10,000 resamples, pooled over seeds), 95% CI.", "",
          "PASS = (a) SNN trained concepts beat random-112 at NEC = 5 (CI > 0) **and** (b) SNN accuracy within 1 point "
          "of ResNet-34 + MLP with the same variant (CI of SNN - ANN > -1).", "",
          "| Variant | SNN acc | ANN acc | SNN - ANN (95% CI) | Concept errors % SNN / ANN | SNN NEC-5 trained vs random-112 "
          "| SNN gap vs random (95% CI) | ANN NEC-5 trained vs random-112 | Verdict |",
          "|---|---|---|---|---|---|---|---|---|"] + rows + [
          "", "base = the published joint runs (lambda_concept 1). Variants: lc5 / lc20 = concept loss weight 5 / 20; "
          "lc5pw = 5 + rare-concept weighting; seq = sequential CBM (concepts first, then a head on frozen concepts)."]
    md[0] += f" (seeds {', '.join(map(str, seeds))})"
    open(rav._safe_path(os.path.join(OUT_ROOT, f"report{REPORT_TAG}.md")), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(J, open(rav._safe_path(os.path.join(OUT_ROOT, f"report{REPORT_TAG}.json")), "w", encoding="utf-8"), indent=1)
    print("\n".join(md))


# ------------------------------------------------------------------------------------------------ main
def main(args):
    set_out(os.path.join(ROOT, "gonogo_step1_dry" if args.dry_run else "gonogo_step1"))
    os.makedirs(OUT_ROOT, exist_ok=True)
    if not args.dry_run:
        sys.stdout = rav._Tee(os.path.join(OUT_ROOT, "log.txt"))
    before = rav.fingerprint_protected()
    rows_all = rav.split_rows()
    rows = rows_all["train_fit"]
    n_concepts = rav.n_concepts_of(rows)
    lab = rav.train_labels(rows)
    pos = lab[0].sum(0)
    CUR["pw"] = torch.from_numpy(np.clip((len(lab[0]) - pos) / np.maximum(pos, 1), 1.0, PW_MAX)).float()
    feats = cam.Feats()
    _fast_nec_path()
    variants = ("lc5", "seq") if args.dry_run else (tuple(args.variants.split(",")) if args.variants else tuple(VARIANTS))
    seeds = (0,) if args.dry_run else (tuple(int(x) for x in args.seeds.split(",")) if args.seeds else SEEDS)
    t0 = time.time()
    if not args.report:
        evs = {k: rav.load_eval(rav.FEATURE_KEY[k], rows_all) for k in MODELS}
        for v in variants:
            for s in seeds:
                for k in MODELS:
                    train_variant(v, k, s, evs[k], lab, n_concepts,
                                  epochs=1 if args.dry_run else rav.EPOCHS, max_batches=5 if args.dry_run else None)
                    print(f"  elapsed {(time.time() - t0) / 60:.1f} min")
        for v in (("base",) + variants if not args.dry_run else variants):
            for s in seeds:
                for k in MODELS:
                    nec_unit(v, k, s, feats, n_concepts)
            print(f"  NEC {v} done, elapsed {(time.time() - t0) / 60:.1f} min")
    if args.dry_run:
        print("[Dry run] training (both variants types), stage 2, finish_run and NEC = 5 all ran. "
              "Output only in gonogo_step1_dry/ (safe to delete).")
        return
    if args.seeds:   # confirmation run on other seeds: separate report, the 3-seed report is kept
        global REPORT_TAG
        REPORT_TAG = "_seeds" + args.seeds.replace(",", "")
    build_report(("base",) + variants, seeds, feats)
    changed = rs.compare_fingerprints(before, rav.fingerprint_protected())
    print(f"\n[Guard] {'WARNING: changed outside gonogo_step1/: ' + str(changed[:10]) if changed else 'PROTECTED FILES UNCHANGED.'}"
          f"  Total {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seeds", default="", help="e.g. 3,4 for a confirmation run (separate report)")
    ap.add_argument("--variants", default="", help="e.g. lc3")
    main(ap.parse_args())
