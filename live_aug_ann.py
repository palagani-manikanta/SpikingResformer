"""
live_aug_ann.py -- ResNet-34 + MLP with LIVE augmentation (review item: "fairness gap in augmentation").

WHY: live_aug_order_test.py trained the spiking GRU CBM with a fresh augmented view of every image in every epoch
(60.06%, 3 seeds), but the ANN controls were only ever trained on the 8 cached views. A reviewer will ask whether
the ANN gains as much from live augmentation. This script trains the capacity-matched ResNet-34 + MLP CBM with
EXACTLY the same live views (same TRAIN_TF parameters for every (seed, epoch, image), drawn by
live_aug_order_test.draw_epoch_params), the same recipe, initialisation and selection rules, and compares it with
the spiking GRU CBM of live_aug_order/ image by image.

STANDALONE: imports live_aug_order_test.py (Run, finish, draw_epoch_params), run_aug_views.py and
run_seeds_extra.py read-only and redirects live_aug_order_test's output paths and model list IN MEMORY to
live_aug_ann/. Nothing that already exists is edited. A write guard keeps every output inside live_aug_ann/ and a
fingerprint of all other repo files is checked at the end.

OUTPUTS: live_aug_ann/runs/seed<s>/ann34_mlp/{best_acc.pth, best_auc.pth, outputs_acc.npz, outputs_auc.npz,
         history.json, result.json}, live_aug_ann/report.md, report.json, log.txt

USAGE (repo root, project venv python, CUDA GPU):
  python live_aug_ann.py --dry-run          # 1 epoch on 64 images, 2 batches; writes nothing; prints s/epoch
  python live_aug_ann.py                    # seeds 0,1,2 (resumable: re-run the same command after a stop)
  python live_aug_ann.py --report           # rebuild the report from finished seeds
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch

import live_aug_order_test as lao
import run_aug_views as rav
import run_seeds as rs
import run_seeds_extra as rse
from train_cbm import DEVICE

KIND = "ann34_mlp"
OUT_ROOT = os.path.join(ROOT, "live_aug_ann")
SNN_LIVE = os.path.join(ROOT, "live_aug_order", "runs")      # the spiking model's live-augmentation runs (read-only)

# ---- redirect live_aug_order_test to this experiment (in memory only) ------------------------------------------
lao.OUT_ROOT = OUT_ROOT
lao.RUNS_DIR = os.path.join(OUT_ROOT, "runs")
lao.PLATT_DIR = os.path.join(OUT_ROOT, "platt")
lao.LOG_PATH = os.path.join(OUT_ROOT, "log.txt")
lao.KINDS = (KIND,)
lao.LABELS = {KIND: "ResNet-34 + MLP (live aug.)"}
_orig_config = lao.config


def _config(kind, seed):
    c = _orig_config(kind, seed)
    c["augmentation"] = c["augmentation"].replace("SNN features float16", "ResNet-34 features float16")
    return c


lao.config = _config


@torch.no_grad()
def live_features_r34(net, rows, p, max_images=None):
    """[n, 512] float16: this epoch's augmented views through the frozen ResNet-34 (batch 32, train_fit order)."""
    rows = rows if max_images is None else rows[:max_images]
    pp = p if max_images is None else {k: v[:, :max_images] for k, v in p.items()}
    loader = torch.utils.data.DataLoader(rav.ViewDataset(rows, pp, 0), batch_size=lao.BATCH, shuffle=False,
                                         num_workers=0)
    out = np.empty((len(rows), 512), np.float16)
    for imgs, idx in loader:
        idx = idx.numpy()
        out[idx[0]:idx[-1] + 1] = net(imgs.to(DEVICE)).float().cpu().numpy().astype(np.float16)
    return out


lao.live_features = live_features_r34


def _load_outputs(path):
    z = np.load(path)
    return z["pred"], z["cs"]


def build_report(seeds, cids):
    """ANN live vs ANN 8-view, and SNN live vs ANN live (paired over test images)."""
    rows, J = [], {"seeds": list(seeds), "per_seed": {}}
    snn_c, ann_c = [], []
    for s in seeds:
        rj = json.load(open(os.path.join(lao.model_dir(s, KIND), "result.json"), encoding="utf-8"))
        a_live = rj["acc"]["test_acc"]
        a_8 = json.load(open(os.path.join(rav.run_dir(KIND, s), "result.json"), encoding="utf-8"))["acc"]["test_acc"]
        sp = os.path.join(SNN_LIVE, f"seed{s}", "learned_decoder")
        s_live = json.load(open(os.path.join(sp, "result.json"), encoding="utf-8"))["acc"]["test_acc"]
        pa, _ = _load_outputs(os.path.join(lao.model_dir(s, KIND), "outputs_acc.npz"))
        ann_c.append(pa == cids)
        so = os.path.join(sp, "outputs_acc.npz")
        if os.path.isfile(so):
            ps, _ = _load_outputs(so)
            snn_c.append(ps == cids)
        J["per_seed"][s] = {"ann34_mlp_live": a_live, "ann34_mlp_8view": a_8, "snn_gru_live": s_live}
        rows.append(f"| {s} | {s_live:.2f} | {a_live:.2f} | {a_8:.2f} | {a_live - a_8:+.2f} |")
    mean = lambda k: float(np.mean([v[k] for v in J["per_seed"].values()]))
    md = ["# ResNet-34 + MLP with live augmentation", "",
          "Same live views (TRAIN_TF parameters per seed, epoch and image), recipe, initialisation and selection "
          "(held-out class accuracy) as the spiking GRU CBM in `live_aug_order/`.", "",
          "| Seed | SNN + GRU, live | ResNet-34 + MLP, live | ResNet-34 + MLP, 8 views | ANN live - 8 views |",
          "|---|---|---|---|---|"] + rows + [
          f"| mean | {mean('snn_gru_live'):.2f} | {mean('ann34_mlp_live'):.2f} | {mean('ann34_mlp_8view'):.2f} | "
          f"{mean('ann34_mlp_live') - mean('ann34_mlp_8view'):+.2f} |", ""]
    if len(snn_c) == len(seeds):
        d = (np.array(snn_c, float) - np.array(ann_c, float)).mean(0) * 100
        rng = np.random.default_rng(20260826)
        m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)])
        lo, hi = np.percentile(m, [2.5, 97.5])
        lo90, hi90 = np.percentile(m, [5, 95])
        J["gap_snn_minus_ann_live"] = {"gap": float(d.mean()), "ci95": [float(lo), float(hi)],
                                       "ci90": [float(lo90), float(hi90)],
                                       "tost_p_margin1": float(max((m <= -1).mean(), (m >= 1).mean()))}
        md.append(f"SNN + GRU minus ResNet-34 + MLP, both with live augmentation: {d.mean():+.2f} points, 95% CI "
                  f"[{lo:+.2f}, {hi:+.2f}] (paired bootstrap over test images, 10,000 resamples, pooled over seeds); "
                  f"TOST p (margin 1 point) = {J['gap_snn_minus_ann_live']['tost_p_margin1']:.3f}.")
    else:
        md.append("SNN per-image outputs (live_aug_order/runs/seed*/learned_decoder/outputs_acc.npz) not found: "
                  "paired test skipped; compare the means above.")
    os.makedirs(OUT_ROOT, exist_ok=True)
    open(lao._safe_path(os.path.join(OUT_ROOT, "report.md")), "w", encoding="utf-8").write("\n".join(md) + "\n")
    lao._write_json(J, os.path.join(OUT_ROOT, "report.json"))
    print("\n".join(md))


def main(args):
    seeds = lao.parse_seeds(args.seeds)
    rows_all = rav.split_rows()
    rows = rows_all["train_fit"]
    n_concepts = rav.n_concepts_of(rows)
    ev = rav.load_eval("r34", rows_all)
    cids = np.asarray(ev["test"][2])
    if args.report:
        build_report(seeds, cids)
        return
    net = rse._resnet(34)[0]
    lab = rav.train_labels(rows)
    if args.dry_run:
        n = 64
        sizes = rav.image_sizes(rows[:n])
        t0 = time.time()
        p = lao.draw_epoch_params(sizes, 0, 1)
        f = live_features_r34(net, rows[:n], p)
        t_f = time.time() - t0
        R = lao.Run(KIND, 0, n_concepts)
        at = torch.from_numpy(lab[0][:n]).float().to(DEVICE)
        yt = torch.from_numpy(lab[1][:n]).long().to(DEVICE)
        val = rs._batches(*rav._dev(*ev["held_out"]))
        R.epoch(1, f, at, yt, val, time.time(), max_batches=2)
        h = R.history
        print(f"[Dry run] features {f.shape} {f.dtype}, {t_f / n * len(rows):.0f} s/epoch for the backbone "
              f"(estimate, {len(rows)} images); 2 training batches ok; held-out acc {h['val_class_acc'][-1]:.2f}% "
              f"(1 epoch on 64 images, expected low). Nothing written.")
        return
    sys.stdout = lao._Tee(lao.LOG_PATH)
    before = lao.fingerprint()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sizes = rav.image_sizes(rows)
    t0 = time.time()
    for i, s in enumerate(seeds):
        if lao.seed_done(s):
            print(f"[Seed {s}] done earlier: SKIPPED")
            continue
        print(f"\n[Seed {s}] ResNet-34 + MLP, {lao.EPOCHS} epochs, live augmentation")
        lao.train_seed(s, net, rows, sizes, lab, ev, n_concepts, t0, len(seeds), i)
    build_report(seeds, cids)
    after = lao.fingerprint()
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    print(f"\n[Guard] {'WARNING: changed outside live_aug_ann/: ' + str(changed[:10]) if changed else 'PROTECTED FILES UNCHANGED: all ' + str(len(before)) + ' pre-existing files verified.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    main(ap.parse_args())
