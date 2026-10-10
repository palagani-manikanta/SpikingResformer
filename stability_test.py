"""
stability_test.py -- how stable are per-image predictions and concepts of the SNN + GRU CBM
under numerically tiny input perturbations? (review item found while building demo_live.py)

For each test image the frozen SpikingResformer-Ti is run live (EVAL_TF, batch of images) on
  * the clean image                                   -> "clean"
  * the image times (1 + eps * N(0, 1)), eps = 1e-6    -> "noise k", k = 1..draws  (about 8 float32 ulps)
and the trained aug_views GRU CBM (seed 0, best held-out class accuracy) + per-concept Platt maps the
spikes to concepts and a class. The saved test features (seeds/cache/test.npz, computed once on the
GPU in batches of 32) give a fourth condition, "saved".

Reported: correlation of the [4 x 1536] spike features with the clean run, class agreement between
conditions, accuracy of each condition, how many of the 112 concept decisions (calibrated p >= 0.5)
change, and the confidence of images whose class changes vs those whose class does not.

Nothing is trained; writes only stability/report.json and stability/report.md.

Usage (inside SpikingResformer-cbm):
  python stability_test.py                     # all 5,794 test images (GPU recommended)
  python stability_test.py --n 500             # 500 random test images
  python stability_test.py --indices-file f.json   # a fixed list of test indices
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from spikingjelly.activation_based import functional

try:
    import cupy  # noqa: F401
except ImportError:                                    # same CPU fallback as demo_live.py
    import spikingjelly.activation_based.base as _sj_base
    _sj_base.check_backend_library = lambda backend: None

import demo_live as dl                                  # model loading + head, identical to the demo

OUT_DIR = os.path.join(ROOT, "stability")


@torch.no_grad()
def spikes(snn, x):
    functional.reset_net(snn.backbone)
    snn.backbone(x)
    return snn._hooked_lif._spike_seq.mean(dim=(-2, -1)).permute(1, 0, 2).float()   # [B, T, C]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="random test images (default: all)")
    p.add_argument("--indices-file", default=None)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0, help="CBM seed")
    p.add_argument("--noise-seed", type=int, default=12345)
    args = p.parse_args()

    data = dl.load_test_cache()
    if args.indices_file:
        idx = np.array(json.load(open(args.indices_file)), dtype=int)
    elif args.n:
        idx = np.sort(np.random.default_rng(2026).choice(dl.N_TEST, args.n, replace=False))
    else:
        idx = np.arange(dl.N_TEST)
    n_c = data["attrs"].shape[1]
    model, platt, epoch, _ = dl.load_cbm(args.seed, n_c)
    snn = dl.SpikingResformerCBM(backbone=dl.build_spiking_backbone(), n_concepts=n_c, n_classes=200,
                                 readout_type="spike_rate", backbone_dim=1536).to(dl.DEVICE).eval()
    gen = torch.Generator().manual_seed(args.noise_seed)
    conds = ["saved", "clean"] + [f"noise{k + 1}" for k in range(args.draws)]
    feats = {c: [] for c in conds}
    t0 = time.time()
    for s in range(0, len(idx), args.batch):
        b = idx[s:s + args.batch]
        x = torch.stack([dl.EVAL_TF(Image.open(os.path.join(dl.IMAGES_DIR, str(data["image_path"][i])))
                                    .convert("RGB")) for i in b])
        feats["saved"].append(torch.from_numpy(data["snn"][b]))
        feats["clean"].append(spikes(snn, x.to(dl.DEVICE)).cpu())
        for k in range(args.draws):
            xn = x * (1 + args.eps * torch.randn(x.shape, generator=gen))
            feats[f"noise{k + 1}"].append(spikes(snn, xn.to(dl.DEVICE)).cpu())
        print(f"  {min(s + args.batch, len(idx))}/{len(idx)} images, {time.time() - t0:.0f} s", flush=True)
    feats = {c: torch.cat(v) for c, v in feats.items()}

    y = data["cids"][idx]
    out = {}
    for c in conds:
        raw, cal, logits = dl.run_head(model, platt, feats[c].to(dl.DEVICE))
        prob = torch.softmax(logits, 1)
        out[c] = {"pred": logits.argmax(1).cpu().numpy(), "conf": prob.max(1).values.cpu().numpy(),
                  "cal": cal.cpu().numpy()}

    def corr(a, b):
        a, b = a.reshape(len(a), -1).numpy(), b.reshape(len(b), -1).numpy()
        return np.array([np.corrcoef(u, v)[0, 1] for u, v in zip(a, b)])

    ref = out["clean"]
    rep = {"n_images": int(len(idx)), "eps": args.eps, "draws": args.draws, "cbm_seed": args.seed,
           "cbm_epoch": epoch, "device": dl.DEVICE, "indices": idx.tolist(), "conditions": {}}
    for c in conds:
        o = out[c]
        flip = o["pred"] != ref["pred"]
        rep["conditions"][c] = {
            "accuracy": float((o["pred"] == y).mean() * 100),
            "spike_feature_corr_vs_clean_mean": float(corr(feats[c], feats["clean"]).mean()),
            "class_agreement_vs_clean": float((~flip).mean() * 100),
            "concept_decisions_changed_vs_clean_mean": float(((o["cal"] >= 0.5) != (ref["cal"] >= 0.5)).sum(1).mean()),
            "concept_prob_abs_change_vs_clean_mean": float(np.abs(o["cal"] - ref["cal"]).mean()),
        }
    noise = [c for c in conds if c.startswith("noise")]
    if len(noise) >= 2:
        rep["noise1_vs_noise2_class_agreement"] = float((out[noise[0]]["pred"] == out[noise[1]]["pred"]).mean() * 100)
    changed = np.zeros(len(idx), bool)
    for c in noise:
        changed |= out[c]["pred"] != ref["pred"]
    conf = ref["conf"]
    rep["any_noise_changed_class_pct"] = float(changed.mean() * 100)
    rep["clean_conf_mean_stable"] = float(conf[~changed].mean()) if (~changed).any() else None
    rep["clean_conf_mean_changed"] = float(conf[changed].mean()) if changed.any() else None
    for thr in (0.5, 0.8):
        m = conf >= thr
        rep[f"changed_pct_when_conf_ge_{thr}"] = float(changed[m].mean() * 100) if m.any() else None
        rep[f"n_conf_ge_{thr}"] = int(m.sum())
        rep[f"changed_pct_when_conf_lt_{thr}"] = float(changed[~m].mean() * 100) if (~m).any() else None
    # binomial 95% CI (Wilson) for the main flip rate
    k, n = int(changed.sum()), len(idx)
    z = 1.96
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    rep["any_noise_changed_class_wilson95"] = [float(100 * (centre - half)), float(100 * (centre + half))]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    lines = [f"# Prediction stability under tiny input noise (eps = {args.eps:g}, {args.draws} draws)", "",
             f"{len(idx)} test images, CBM seed {args.seed} (epoch {epoch}), device {dl.DEVICE}.", "",
             "| Condition | Accuracy (%) | Spike-feature r vs clean | Class agreement vs clean (%) | "
             "Concept decisions changed (of 112) |", "|:---|:---:|:---:|:---:|:---:|"]
    for c in conds:
        r = rep["conditions"][c]
        lines.append(f"| {c} | {r['accuracy']:.2f} | {r['spike_feature_corr_vs_clean_mean']:.3f} | "
                     f"{r['class_agreement_vs_clean']:.1f} | {r['concept_decisions_changed_vs_clean_mean']:.2f} |")
    lines += ["", f"Class changed under at least one noise draw: {rep['any_noise_changed_class_pct']:.1f}% "
              f"(95% CI {rep['any_noise_changed_class_wilson95'][0]:.1f}-{rep['any_noise_changed_class_wilson95'][1]:.1f}%).",
              f"Mean clean top-class probability: {rep['clean_conf_mean_stable']:.3f} (class stable) vs "
              f"{rep['clean_conf_mean_changed']:.3f} (class changed).",
              f"Changed when clean confidence >= 0.5: {rep['changed_pct_when_conf_ge_0.5']:.1f}% "
              f"(n = {rep['n_conf_ge_0.5']}); < 0.5: {rep['changed_pct_when_conf_lt_0.5']:.1f}%."]
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
