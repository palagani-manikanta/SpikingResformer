"""
improved_analyses.py -- the paper's analyses repeated on the IMPROVED model (lambda_c = 3 + attention readout).

WHY: the abstract now reports the improved model (gonogo_step2: SNN 61.16%, ResNet-34 + MLP 60.29%), but
calibration, intervention, the full NEC curve and the energy estimate were measured on the main model
(lambda_c = 1, mean pooling). This script repeats them, with the SAME protocols, on the 5 + 5 improved models:

  A. Calibration: per-concept Platt scaling fitted on the 899 held-out concept logits (calibration_all_models.
     fit_platt_params, lambda 5), test ECE raw -> calibrated, concept AUC.
  B. Intervention: a fresh class head per arm trained on the frozen concepts (raw / global Platt / per-concept
     Platt) with ablation_calibrated_head.train_head (50 epochs, same recipe), then the same shared random concept
     subsets replaced by true values (fractions 0-100%).
  C. NEC curve: VLG-CBM sparse layer at NEC = 5..30 (full 50-point path, nec_sparse_head.nec_evaluate), trained
     concepts and random-112 on the same features; ANEC-5 and ANEC-avg.
  D. Energy of the attention readout: measured spike counts at the tapped layer on all test images
     -> accumulates (8 heads score every spike + 1 pooling add), plus the softmax; ResNet-34 analytic.

Reads gonogo_step2/runs/*/best_acc.pth and spatial_cache/ (read-only). Writes only improved_analyses/.
USAGE (inside New folder\\SpikingResformer-cbm, GPU):  python improved_analyses.py   (resumable per unit)
                                                       python improved_analyses.py --report
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch

import gonogo_step2 as g2
import run_aug_views as rav
import run_seeds as rs
import nec_sparse_head as nsh
import calibration_all_models as cam
import ablation_calibrated_head as abl
from train_mlp_notime import _mean_auc
from train_cbm import DEVICE

OUT = os.path.join(ROOT, "improved_analyses")
MODELS = g2.MODELS
SEEDS = g2.SEEDS
LABEL = {"learned_decoder": "SNN + GRU + attention", "ann34_mlp": "ResNet-34 + MLP + attention"}
E_MAC, E_AC = 4.6, 0.9                       # pJ, as energy_audit_v2
BASE_MJ = {"learned_decoder": 2.978, "ann34_mlp": 16.860}     # energy_audit_v2: SNN stem once; ResNet-34 + MLP
BOOT_SEED, N_BOOT = 20260826, 10_000


def guard(p):
    rp, root = os.path.realpath(p), os.path.realpath(OUT)
    if not (rp == root or rp.startswith(root + os.sep)):
        raise RuntimeError(f"[Guard] refusing to write outside {OUT}: {p}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return p


def labels():
    out = {}
    for s in ("train_fit", "held_out", "test"):
        z = np.load(os.path.join(rs.CACHE_DIR, f"{s}.npz"))
        out[s] = (z["attrs"].astype(np.float32), z["cids"].astype(np.int64))
    return out


def load(kind, seed, n_concepts):
    ck = torch.load(os.path.join(g2.run_dir(kind, seed), "best_acc.pth"), map_location="cpu", weights_only=False)
    m = g2.AttCBM(kind, n_concepts).to(DEVICE)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["result"]


def unit(kind, seed, lab, n_concepts):
    p = os.path.join(OUT, "units", f"{kind}_seed{seed}.json")
    if os.path.isfile(p):
        return json.load(open(p, encoding="utf-8"))
    t0 = time.time()
    model, res = load(kind, seed, n_concepts)
    fe = g2.Feats(kind)
    H, Z = {}, {}
    for s in ("train_fit", "held_out", "test"):
        H[s], Z[s] = g2.reps(model, fe, fe.split(s))
    (_, yho), (ate, yte) = lab["held_out"], lab["test"]
    atr, ytr = lab["train_fit"]
    aho = lab["held_out"][0]
    # A. calibration
    pl = cam.fit_platt_params(Z["held_out"], aho)
    raw = cam.apply_platt(Z["test"], pl, "raw"); cal = cam.apply_platt(Z["test"], pl, "per_concept_platt")
    A = {"auc": float(_mean_auc(raw, ate)), "ece_raw": cam.mean_ece(raw, ate), "ece_cal": cam.mean_ece(cal, ate)}
    print(f"  [A] {LABEL[kind]} seed {seed}: AUC {A['auc']:.4f}  ECE {A['ece_raw']:.4f} -> {A['ece_cal']:.4f}")
    # B. intervention, head retrained per arm
    abp = cam.abl_platt(pl)
    subsets = abl.make_intervention_subsets(n_concepts)
    Ctr, Cte = torch.from_numpy(atr).float(), torch.from_numpy(ate).float()
    B = {}
    for arm in abl.ARMS:
        Xtr, Xte = abl.arm_inputs(Z["train_fit"], arm, abp), abl.arm_inputs(Z["test"], arm, abp)
        head = abl.train_head(Xtr, Ctr, torch.from_numpy(ytr).long(), seed=seed, **{k: v for k, v in
                              cam.ABL_RECIPE.items() if k != "batch_size"}, batch_size=cam.ABL_RECIPE["batch_size"])
        sweep = abl.intervention_sweep(head, Xte, Cte, yte, subsets)
        i25 = cam.icrc_per_image(head, Xte, Cte, yte, subsets[0.25])
        B[arm] = {"test_acc": abl.accuracy(abl.predict(head, Xte), yte),
                  "icrc": {str(f): float(np.mean(v)) for f, v in sweep.items()}, "icrc25_per_image": i25.tolist()}
        print(f"  [B] {LABEL[kind]} seed {seed} {arm:<17s} acc {B[arm]['test_acc']:.2f}%  ICRC@25% {B[arm]['icrc']['0.25']:.2f}%")
    # C. full NEC curve
    C = {}
    g = torch.Generator().manual_seed(nsh.RANDOM_SEED_BASE + 1000 * seed + sum(map(ord, kind)))
    Wr = torch.randn(112, H["train_fit"].shape[1], generator=g, dtype=torch.float64).numpy()
    for name, X in (("trained", Z), ("random112", {s: H[s].astype(np.float64) @ Wr.T for s in H})):
        r = nsh.nec_evaluate(X["train_fit"], ytr, {"held_out": (X["held_out"], yho), "test": (X["test"], yte)},
                             200, targets=nsh.NEC_TARGETS, device=DEVICE, dtype=torch.float32)
        C[name] = {str(k): r["targets"][k]["test_acc"] for k in nsh.NEC_TARGETS}
        C[name]["test_correct5"] = (r["preds"][5]["test"] == yte).tolist()
        print(f"  [C] {LABEL[kind]} seed {seed} {name}: ANEC-5 {C[name]['5']:.2f}  "
              f"ANEC-avg {np.mean([C[name][str(k)] for k in nsh.NEC_TARGETS]):.2f}")
    out = {"model": kind, "seed": seed, "test_acc": res["test_acc"], "A": A, "B": B, "C": C, "seconds": time.time() - t0}
    json.dump(out, open(guard(p), "w"))
    return out


def energy():
    """Attention readout cost from measured test spike counts (SNN) and analytically (ResNet-34)."""
    p = os.path.join(OUT, "energy.json")
    if os.path.isfile(p):
        return json.load(open(p))
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
    arr = np.load(g2.cpath("test", "snn"), mmap_mode="r")
    tot = 0
    for i in range(0, len(arr), 256):
        tot += int(lut[np.asarray(arr[i:i + 256])].sum())
    spikes = tot / len(arr)                                   # spikes per image at the tap (T x C x 196)
    H, T, P = g2.HEADS, g2.T, g2.P_SNN
    snn_ac = spikes * (H + 1)                                 # scoring (H heads) + weighted pooling (1 add per spike)
    snn_mac = 2 * H * P * T                                   # softmax: exp + normalisation, counted as MACs
    snn_mj = (snn_ac * E_AC + snn_mac * E_MAC) * 1e-9
    ann_mac = g2.C_ANN * g2.P_ANN * H + g2.C_ANN * g2.P_ANN + 2 * H * g2.P_ANN
    ann_mj = ann_mac * E_MAC * 1e-9
    e = {"spikes_per_image": spikes, "spike_density": spikes / (T * g2.C_SNN * P), "snn_att_ac": snn_ac,
         "snn_att_mac": snn_mac, "snn_att_mj": snn_mj, "ann_att_mac": ann_mac, "ann_att_mj": ann_mj,
         "snn_total_mj": BASE_MJ["learned_decoder"] + snn_mj, "ann_total_mj": BASE_MJ["ann34_mlp"] + ann_mj,
         "ratio": (BASE_MJ["ann34_mlp"] + ann_mj) / (BASE_MJ["learned_decoder"] + snn_mj),
         "extra_bytes_spike_map_read": T * g2.C_SNN * P / 8}
    json.dump(e, open(guard(p), "w"), indent=1)
    print(f"  [D] spikes/image at tap {spikes:,.0f} (density {e['spike_density']:.3f}); attention readout "
          f"{snn_mj * 1000:.2f} uJ; SNN total {e['snn_total_mj']:.3f} mJ vs ResNet-34 + MLP {e['ann_total_mj']:.3f} mJ "
          f"-> {e['ratio']:.2f}x")
    return e


def boot(d):
    rng = np.random.default_rng(BOOT_SEED); n = len(d)
    m = np.concatenate([d[rng.integers(0, n, size=(1000, n))].mean(1) for _ in range(N_BOOT // 1000)])
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def report():
    U = {k: [json.load(open(os.path.join(OUT, "units", f"{k}_seed{s}.json"))) for s in SEEDS] for k in MODELS}
    e = energy()
    ms = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}"
    mp = lambda v: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
    J, rows = {"energy": e, "models": {}}, []
    for k in MODELS:
        u = U[k]
        r = {"acc": [x["test_acc"] for x in u], "auc": [x["A"]["auc"] for x in u],
             "ece_raw": [x["A"]["ece_raw"] for x in u], "ece_cal": [x["A"]["ece_cal"] for x in u],
             "head_raw_acc": [x["B"]["raw"]["test_acc"] for x in u],
             "head_cal_acc": [x["B"]["per_concept_platt"]["test_acc"] for x in u],
             "icrc": {arm: {f: [x["B"][arm]["icrc"][f] for x in u] for f in u[0]["B"][arm]["icrc"]} for arm in abl.ARMS},
             "nec_trained": {kk: [x["C"]["trained"][kk] for x in u] for kk in map(str, nsh.NEC_TARGETS)},
             "nec_random": {kk: [x["C"]["random112"][kk] for x in u] for kk in map(str, nsh.NEC_TARGETS)}}
        avg = lambda d: [np.mean([d[str(t)][i] for t in nsh.NEC_TARGETS]) for i in range(len(SEEDS))]
        r["anec_avg_trained"], r["anec_avg_random"] = avg(r["nec_trained"]), avg(r["nec_random"])
        r["icrc25_gain_cal_minus_raw"] = boot(np.mean([np.array(x["B"]["per_concept_platt"]["icrc25_per_image"]) -
                                                       np.array(x["B"]["raw"]["icrc25_per_image"]) for x in u], 0) * 100)
        r["nec5_trained_minus_random"] = boot(np.mean([np.array(x["C"]["trained"]["test_correct5"], float) -
                                                       np.array(x["C"]["random112"]["test_correct5"], float) for x in u], 0) * 100)
        J["models"][k] = r
        rows.append(f"| {LABEL[k]} | {mp(r['acc'])} | {ms(r['auc'])} | {ms(r['ece_raw'])} -> {ms(r['ece_cal'])} | "
                    f"{np.mean(r['icrc']['raw']['0.25']):.2f} -> {np.mean(r['icrc']['per_concept_platt']['0.25']):.2f} "
                    f"({r['icrc25_gain_cal_minus_raw'][0]:+.2f} [{r['icrc25_gain_cal_minus_raw'][1]:+.2f}, {r['icrc25_gain_cal_minus_raw'][2]:+.2f}]) | "
                    f"{np.mean(r['nec_trained']['5']):.2f} vs {np.mean(r['nec_random']['5']):.2f} | "
                    f"{np.mean(r['anec_avg_trained']):.2f} vs {np.mean(r['anec_avg_random']):.2f} |")
    s, a = J["models"]["learned_decoder"], J["models"]["ann34_mlp"]
    d25 = boot(np.mean([np.array(x["B"]["per_concept_platt"]["icrc25_per_image"]) for x in U["learned_decoder"]], 0) * 100 -
               np.mean([np.array(x["B"]["per_concept_platt"]["icrc25_per_image"]) for x in U["ann34_mlp"]], 0) * 100)
    J["icrc25_cal_snn_minus_ann"] = d25
    fr = list(s["icrc"]["raw"].keys())
    md = ["# Analyses repeated on the improved model (lambda_c = 3 + attention readout, 5 seeds)", "",
          "| Model | Accuracy | Concept AUC | ECE raw -> per-concept Platt | ICRC@25% raw -> calibrated head (gain, 95% CI) "
          "| ANEC-5 trained vs random-112 | ANEC-avg trained vs random-112 |", "|---|---|---|---|---|---|---|"] + rows + [
          "", f"Calibrated-head ICRC@25%, SNN - ANN: {d25[0]:+.2f} [{d25[1]:+.2f}, {d25[2]:+.2f}] points.", "",
          "Intervention curve (mean over seeds, head retrained per arm), fractions " + ", ".join(fr) + ":"]
    for k, r in J["models"].items():
        for arm in ("raw", "per_concept_platt"):
            md.append(f"- {LABEL[k]}, {arm}: " + ", ".join(f"{np.mean(r['icrc'][arm][f]):.2f}" for f in fr))
    md += ["", "NEC curve (mean test accuracy at NEC = 5..30):"]
    for k, r in J["models"].items():
        for nm, key in (("trained", "nec_trained"), ("random-112", "nec_random")):
            md.append(f"- {LABEL[k]}, {nm}: " + ", ".join(f"{np.mean(r[key][str(t)]):.2f}" for t in nsh.NEC_TARGETS))
    md += ["", f"Energy: attention readout {e['snn_att_mj'] * 1000:.2f} uJ per image ({e['snn_att_ac']:,.0f} AC + "
           f"{e['snn_att_mac']:,} MAC; tap spike density {e['spike_density']:.3f}) vs {e['ann_att_mj'] * 1000:.2f} uJ for "
           f"ResNet-34; totals {e['snn_total_mj']:.3f} vs {e['ann_total_mj']:.3f} mJ -> {e['ratio']:.2f}x (compute only, "
           "45 nm, same rule as energy_audit_v2)."]
    open(guard(os.path.join(OUT, "report.md")), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(J, open(guard(os.path.join(OUT, "report.json")), "w"), indent=1, default=float)
    print("\n".join(md))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    if not a.report:
        rows_all = rav.split_rows()
        n_concepts = rav.n_concepts_of(rows_all["train_fit"])
        lab = labels()
        t0 = time.time()
        for s in SEEDS:
            for k in MODELS:
                unit(k, s, lab, n_concepts)
                print(f"  elapsed {(time.time() - t0) / 60:.1f} min")
        energy()
    report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    main(ap.parse_args())
