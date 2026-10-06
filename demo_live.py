"""
demo_live.py -- live demo of the trained model, one step at a time (Team 07).

What it shows for ONE bird photo:
  STEP 1  the frozen SpikingResformer-Ti (base paper 1) looks at the photo for T = 4 time steps
  STEP 2  spikes (only 0 or 1) recorded at layer layers.2.6.down.0 -> 4 x 1536 numbers
  STEP 3  the GRU decoder reads t1 -> t2 -> t3 -> t4 in order -> 1536 numbers
  STEP 4  the concept layer -> 112 concept probabilities, Platt-calibrated
          (a, b fitted on the 899 held-out photos only, saved by calibration_all_models.py)
  STEP 5  the class head, which sees ONLY the 112 concepts -> bird species
  STEP 6  simulated correction: the dataset's true concept labels play the human and fix the
          concepts the model got most wrong, one at a time -> new species
  CHECK   the same photo's saved features (used for every slide number) must match the live run

NOTHING IS TRAINED OR WRITTEN except one picture in demo_output/. It only loads the saved files that
produced the slide numbers:
  aug_views/runs/learned_decoder_seed{0,1,2}/best_acc.pth      trained GRU + concept layer + head
  calibration_all/units/learned_decoder_seed{0,1,2}_acc.json    Platt a, b per concept
  seeds/cache/test.npz                                          saved test features (5,794 photos)

Usage (PowerShell, inside the SpikingResformer-cbm folder):
  python demo_live.py                       a random test photo
  python demo_live.py --index 120           test photo number 120 (0 .. 5793)
  python demo_live.py --wrong               a random test photo the model gets WRONG (good for step 6)
  python demo_live.py --image C:\\path\\bird.jpg   any photo (no true labels -> no step 6, no check)
  python demo_live.py --eval                whole test set from the saved features, all 3 seeds:
                                            58.77 / 58.37 / 58.65 -> mean 58.60%, ECE 0.08 -> 0.02
Options: --seed 0|1|2 (default 0), --fix N (concepts to fix in step 6, default 8), --open (open the figure)
"""
import argparse, json, os, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from spikingjelly.activation_based import functional

try:                                                         # CuPy is only the GPU speed-up for the LIF neurons.
    import cupy  # noqa: F401                                # Without it (a laptop with no GPU), let the model be
except ImportError:                                          # built anyway: build_spiking_backbone() switches
    import spikingjelly.activation_based.base as _sj_base    # every neuron to backend "torch" right after, which
    _sj_base.check_backend_library = lambda backend: None    # gives the same spikes on the CPU.

import run_seeds as rs                                   # CachedCBM: the exact model class the results used
from train_cbm import CSV_PATH, IMAGES_DIR, DEVICE
from train_mlp_notime import EVAL_TF, _mean_ece          # same test transform and ECE as the reports
from anec5_gap_test import build_spiking_backbone
from models.cbm import SpikingResformerCBM

RUNS_DIR = os.path.join(ROOT, "aug_views", "runs")
PLATT_DIR = os.path.join(ROOT, "calibration_all", "units")
TEST_CACHE = os.path.join(ROOT, "seeds", "cache", "test.npz")
OUT_DIR = os.path.join(ROOT, "demo_output")
HOOK_LAYER = "layers.2.6.down.0"
N_TEST = 5794


# ----------------------------------------------------------------------------- loading
def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                                        # older torch has no weights_only
        return torch.load(path, map_location="cpu")


def concept_names():
    with open(CSV_PATH, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    keys = [k for k in header if k not in ("image_id", "image_path", "class_id", "split")]

    def pretty(k):                                           # has_wing_color::grey -> wing color: grey
        part, _, value = k.partition("::")
        return f"{part.replace('has_', '').replace('_', ' ')}: {value.replace('_', ' ')}"
    return [pretty(k) for k in keys]


def class_names():
    path = os.path.join(os.path.dirname(CSV_PATH), "classes.txt")
    names = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            i, folder = line.split()
            names[int(i) - 1] = folder.split(".", 1)[1].replace("_", " ")
    return names


def load_cbm(seed, n_concepts):
    path = os.path.join(RUNS_DIR, f"learned_decoder_seed{seed}", "best_acc.pth")
    ck = _torch_load(path)
    model = rs.CachedCBM("learned_decoder", n_concepts).to(DEVICE).eval()
    model.load_state_dict(ck["model_state"])
    with open(os.path.join(PLATT_DIR, f"learned_decoder_seed{seed}_acc.json"), encoding="utf-8") as f:
        p = json.load(f)["platt"]
    a = torch.tensor(p["a"], dtype=torch.float64, device=DEVICE)
    b = torch.tensor(p["b"], dtype=torch.float64, device=DEVICE)
    return model, (a, b), ck.get("epoch"), path


def load_test_cache():
    z = np.load(TEST_CACHE)
    return {k: z[k] for k in ("snn", "attrs", "cids", "image_path")}


# ----------------------------------------------------------------------------- the model, step by step
@torch.no_grad()
def run_head(model, platt, per_t):
    """per_t [B, 4, 1536] -> raw concept scores, calibrated concept probs, class logits."""
    feat = model.features(per_t)                           # STEP 3: GRU over t1..t4
    z = model.cbl.linear(feat)                             # STEP 4: concept logits
    raw = torch.sigmoid(z)                                 #         what the class head was trained on
    a, b = platt
    cal = torch.sigmoid(a * z.double() + b)                #         Platt: sigmoid(a * logit + b)
    logits = model.head(raw)                               # STEP 5: class from concepts only
    return raw, cal.float(), logits


@torch.no_grad()
def live_spikes(snn, img_path):
    x = EVAL_TF(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    functional.reset_net(snn.backbone)
    t0 = time.time()
    snn.backbone(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    seq = snn._hooked_lif._spike_seq                       # [T, 1, 1536, H, W], values 0/1
    per_t = seq.mean(dim=(-2, -1)).permute(1, 0, 2)        # [1, T, 1536]  (same as the saved cache)
    return seq[:, 0].float().cpu(), per_t.float(), secs


# ----------------------------------------------------------------------------- printing helpers
def line(ch="-"):
    print(ch * 78)


def topk_species(logits, names, k=5):
    p = torch.softmax(logits[0], dim=0).cpu().numpy()
    idx = np.argsort(-p)[:k]
    return [(int(i), names[int(i)], float(p[i])) for i in idx]


def eval_mode(args):
    names = concept_names()
    data = load_test_cache()
    x = torch.from_numpy(data["snn"]).to(DEVICE)
    y = data["cids"]
    attrs = data["attrs"].astype(np.float64)
    line("=")
    print(" WHOLE TEST SET from the saved features: 5,794 photos never used in training")
    line("=")
    print(f" {'seed':<5}{'epoch':>6}{'test acc':>11}{'saved result':>14}{'ECE raw':>10}{'ECE calibrated':>16}")
    accs = []
    for seed in (0, 1, 2):
        model, platt, epoch, _ = load_cbm(seed, len(names))
        preds, cals, raws = [], [], []
        for i in range(0, len(y), 512):
            raw, cal, logits = run_head(model, platt, x[i:i + 512])
            preds.append(logits.argmax(1).cpu().numpy()); cals.append(cal.cpu().numpy()); raws.append(raw.cpu().numpy())
        pred, cal, raw = np.concatenate(preds), np.concatenate(cals), np.concatenate(raws)
        acc = 100.0 * float((pred == y).mean())
        with open(os.path.join(RUNS_DIR, f"learned_decoder_seed{seed}", "result.json"), encoding="utf-8") as f:
            saved = json.load(f)["acc"]["test_acc"]
        accs.append(acc)
        print(f" {seed:<5}{epoch:>6}{acc:>10.2f}%{saved:>13.2f}%{_mean_ece(raw, attrs):>10.3f}{_mean_ece(cal, attrs):>16.3f}")
    print(f" {'mean':<11}{np.mean(accs):>10.2f}%   (slides: 58.60%)")
    line()
    print(" Calibration changes only the concept probabilities, so accuracy is the same before and after.")


def demo_mode(args):
    cnames, snames = concept_names(), class_names()
    n_c = len(cnames)
    model, platt, epoch, ckpt_path = load_cbm(args.seed, n_c)

    data, idx, truth_attrs, truth_cls = None, None, None, None
    if args.image:
        img_path = args.image
        title = os.path.basename(img_path)
    else:
        data = load_test_cache()
        if args.index is not None:
            idx = args.index
        else:
            rng = np.random.default_rng()
            pool = np.arange(N_TEST)
            if args.wrong:                                 # photos the saved run got wrong with >= 50% confidence
                x = torch.from_numpy(data["snn"]).to(DEVICE)
                lg = torch.cat([run_head(model, platt, x[i:i + 512])[2] for i in range(0, N_TEST, 512)])
                p = torch.softmax(lg, 1).max(1)
                pred, conf = p.indices.cpu().numpy(), p.values.cpu().numpy()
                pool = np.flatnonzero((pred != data["cids"]) & (conf >= 0.5))
            idx = int(rng.choice(pool))
        rel = str(data["image_path"][idx])
        img_path = os.path.join(IMAGES_DIR, rel)
        truth_attrs = data["attrs"][idx].astype(np.float32)
        truth_cls = int(data["cids"][idx])
        title = f"test photo #{idx} of {N_TEST}"

    line("=")
    print(" LIVE DEMO  |  calibrated concept bottleneck on a frozen spiking backbone")
    line("=")
    print(f" Photo : {title}")
    print(f"         {img_path}")
    if truth_cls is not None:
        print(f" Truth : {snames[truth_cls]}")
    print(f" Model : {os.path.relpath(ckpt_path, ROOT)} (seed {args.seed}, epoch {epoch}), device {DEVICE}")
    line()

    # STEP 1-2: the frozen spiking backbone, live
    t0 = time.time()
    snn = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=n_c, n_classes=200,
                              readout_type="spike_rate", backbone_dim=rs.SNN_DIM).to(DEVICE).eval()
    n_backbone = sum(p.numel() for p in snn.backbone.parameters())
    n_trainable = sum(p.numel() for p in snn.backbone.parameters() if p.requires_grad)
    load_s = time.time() - t0
    seq, per_t, fwd_s = live_spikes(snn, img_path)
    T, C, H, W = seq.shape
    print(f"STEP 1  Frozen SpikingResformer-Ti (base paper 1) looks at the photo for T = {T} time steps")
    print(f"        {n_backbone / 1e6:.1f}M weights loaded from the authors' ImageNet checkpoint, "
          f"trainable: {n_trainable} (frozen)")
    print(f"        load {load_s:.1f} s, one forward pass {fwd_s:.2f} s")
    vals = sorted({float(v) for v in torch.unique(seq).tolist()})
    rates = [100.0 * float(seq[t].mean()) for t in range(T)]
    print(f"STEP 2  Spikes recorded at layer {HOOK_LAYER}: [{T} time steps x {C} channels x {H} x {W}]")
    print(f"        values in the spike tensor: {vals}  (spikes are only 0 or 1)")
    print("        neurons firing:  " + "   ".join(f"t{t + 1} {r:.1f}%" for t, r in enumerate(rates)))
    print(f"        averaged over the {H} x {W} grid -> {T} x {C} numbers per photo")

    raw, cal, logits = run_head(model, platt, per_t.to(DEVICE))
    raw0, cal0 = raw[0].cpu().numpy(), cal[0].cpu().numpy()
    print(f"STEP 3  GRU decoder reads t1 -> t2 -> t3 -> t4 in order -> {C} numbers")
    print(f"STEP 4  Concept layer -> {n_c} concept probabilities "
          f"(Platt-calibrated, a and b fitted on 899 held-out photos only)")
    order = np.argsort(-cal0)[:args.top]
    hdr = f"        {'concepts the model is most sure about':<44}{'model':>7}"
    print(hdr + (f"{'truth':>8}" if truth_attrs is not None else ""))
    for j in order:
        row = f"          {cnames[j]:<42}{cal0[j]:>7.2f}"
        if truth_attrs is not None:
            ok = (cal0[j] >= 0.5) == (truth_attrs[j] >= 0.5)
            row += f"{'yes' if truth_attrs[j] else 'no':>8}   {'ok' if ok else 'WRONG'}"
        print(row)
    if truth_attrs is not None:
        n_ok = int(((cal0 >= 0.5) == (truth_attrs >= 0.5)).sum())
        print(f"        concepts correct (yes/no at 0.5): {n_ok} of {n_c}")

    top = topk_species(logits, snames)
    print(f"STEP 5  Class head uses ONLY these {n_c} concepts -> species")
    for r, (ci, name, p) in enumerate(top, 1):
        mark = "   <- truth" if ci == truth_cls else ""
        print(f"          {r}. {name:<34}{100 * p:>6.1f}%{mark}")
    if truth_cls is not None:
        print("        prediction is " + ("CORRECT" if top[0][0] == truth_cls else "WRONG"))

    steps = []
    if truth_attrs is not None and top[0][0] == truth_cls:
        print("STEP 6  Nothing to correct: the model is already right (run with --wrong to see a correction)")
    elif truth_attrs is not None and args.fix > 0:
        print("STEP 6  Simulated human correction: true concept labels from the dataset act as the human")
        print("        and fix the concepts the model got most wrong, one at a time")
        print(f"        {'fixed':>5}  {'concept':<36}{'model said':>11}{'truth':>7}   prediction after the fix")
        worst = np.argsort(-np.abs(cal0 - truth_attrs))[:args.fix]
        fixed = raw.clone()
        first_right = None
        print(f"        {0:>5}  {'(none)':<36}{'':>11}{'':>7}   {top[0][1]}")
        for k, j in enumerate(worst, 1):
            fixed[0, j] = float(truth_attrs[j])
            with torch.no_grad():
                pr = topk_species(model.head(fixed), snames, 1)[0]
            right = pr[0] == truth_cls
            if right and first_right is None:
                first_right = k
            steps.append((k, cnames[j], float(cal0[j]), int(truth_attrs[j]), pr[1], right))
            print(f"        {k:>5}  {cnames[j]:<36}{cal0[j]:>11.2f}{'yes' if truth_attrs[j] else 'no':>7}   "
                  f"{pr[1]}{'  (correct)' if right else ''}")
            if right:
                break
        if first_right:
            print(f"        -> correct after fixing {first_right} concept(s)")
        else:
            print(f"        -> still wrong after {args.fix} fixes; try --fix 15")

    if idx is not None:
        cached = torch.from_numpy(data["snn"][idx:idx + 1]).to(DEVICE)
        corr = float(np.corrcoef(cached.cpu().numpy().ravel(), per_t.cpu().numpy().ravel())[0, 1])
        saved = topk_species(run_head(model, platt, cached)[2], snames, 1)[0]
        print(f"CHECK   Saved run for this photo (part of the 58.60% result): {saved[1]} ({100 * saved[2]:.1f}%, "
              f"{'correct' if saved[0] == truth_cls else 'wrong'})")
        print(f"        live spikes vs saved spikes: correlation {corr:.2f}")
        if saved[0] != top[0][0]:
            print("        The live answer differs: deep spiking layers flip some spikes under tiny rounding")
            print("        differences (GPU vs CPU, batch size), which can change an UNSURE prediction.")
    line()

    if not args.no_fig:
        path = save_figure(img_path, title, seq, rates, cnames, cal0, truth_attrs, top, snames, truth_cls,
                           steps, args, idx)
        print(f" Figure saved: {os.path.relpath(path, ROOT)}")
        if args.open and hasattr(os, "startfile"):
            os.startfile(path)


# ----------------------------------------------------------------------------- figure
def save_figure(img_path, title, seq, rates, cnames, cal0, truth_attrs, top, snames, truth_cls, steps, args, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, LIGHT, WRONG = "#2F2F2F", "#6B6B6B", "#B5B5B5", "#C2410C"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LIGHT,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(16, 9), dpi=110, facecolor="white")
    head = dict(fontsize=12.5, color=INK, fontweight="bold", ha="left", va="bottom")
    fig.text(0.03, 0.955, "Live demo: photo -> frozen spiking backbone -> GRU -> 112 concepts -> species",
             fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.03, 0.915, f"{title}" + (f"   |   truth: {snames[truth_cls]}" if truth_cls is not None else "")
             + f"   |   model seed {args.seed}", fontsize=10.5, color=MUTED, va="center")

    # 1. photo (top left)
    fig.text(0.03, 0.865, "1. Input photo", **head)
    ax = fig.add_axes([0.03, 0.50, 0.30, 0.355])
    ax.imshow(Image.open(img_path).convert("RGB"))
    ax.set_axis_off()

    # 2. spikes of the most active channel at t1..t4 (top right)
    ch = int(seq.mean(dim=(0, 2, 3)).argmax())
    fig.text(0.40, 0.865, "2. Spikes of one channel at the 4 time steps (black = 1, white = 0)", **head)
    w = 0.115
    h = w * 16 / 9
    for t in range(seq.shape[0]):
        x0 = 0.40 + t * 0.14
        a = fig.add_axes([x0, 0.62, w, h])
        a.imshow(seq[t, ch].numpy(), cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(MUTED)
        a.set_title(f"t{t + 1}", fontsize=11, color=INK)
        if t < seq.shape[0] - 1:
            fig.text(x0 + w + 0.0125, 0.62 + h / 2, "->", fontsize=13, color=MUTED, ha="center", va="center")
    fig.text(0.40, 0.575, "neurons firing in the whole layer:   " +
             "    ".join(f"t{t + 1}  {r:.1f}%" for t, r in enumerate(rates)), fontsize=10.5, color=MUTED)

    # 3. concepts (bottom left)
    fig.text(0.03, 0.455, "3. Top 10 of 112 concepts (calibrated probability)", **head)
    a = fig.add_axes([0.235, 0.11, 0.235, 0.335])
    order = np.argsort(-cal0)[:10][::-1]
    labels, cols = [], []
    for j in order:
        wrong = truth_attrs is not None and ((cal0[j] >= 0.5) != (truth_attrs[j] >= 0.5))
        labels.append(cnames[j] + ("  (wrong)" if wrong else ""))
        cols.append(WRONG if wrong else INK)
    a.barh(range(len(order)), cal0[order], color=cols, height=0.62)
    a.set_yticks(range(len(order)))
    a.set_yticklabels(labels, fontsize=9.5, color=INK)
    for i, j in enumerate(order):
        a.text(cal0[j] + 0.015, i, f"{cal0[j]:.2f}", va="center", fontsize=9, color=INK)
    a.set_xlim(0, 1.15)
    a.axvline(0.5, color=LIGHT, lw=1, ls="--")
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("probability that the concept is present")

    # 4. species (bottom right)
    fig.text(0.52, 0.455, "4. Species, decided from the 112 concepts only", **head)
    a = fig.add_axes([0.70, 0.11, 0.21, 0.335])
    names = [n for _, n, _ in top][::-1]
    probs = [100 * p for _, _, p in top][::-1]
    ids = [ci for ci, _, _ in top][::-1]
    if truth_cls is None:
        cols = [INK if i == len(names) - 1 else LIGHT for i in range(len(names))]
    else:
        cols = [INK if ci == truth_cls else (WRONG if ci == top[0][0] else LIGHT) for ci in ids]
    a.barh(range(len(names)), probs, color=cols, height=0.62)
    a.set_yticks(range(len(names)))
    a.set_yticklabels([n + ("  (truth)" if ci == truth_cls else "") for ci, n in zip(ids, names)],
                      fontsize=9.5, color=INK)
    for i, p in enumerate(probs):
        a.text(p + 1, i, f"{p:.1f}%", va="center", fontsize=9, color=INK)
    a.set_xlim(0, max(probs) * 1.3 + 1)
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("probability (%)")
    if steps:
        k_ok = next((s[0] for s in steps if s[5]), None)
        msg = (f"Simulated correction: right after fixing {k_ok} concept(s) -> {next(s[4] for s in steps if s[5])}"
               if k_ok else f"Simulated correction: still wrong after fixing {len(steps)} concepts")
        fig.text(0.52, 0.012, msg, fontsize=10.5, color=INK)

    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"demo_test{idx}.png" if idx is not None else f"demo_{os.path.splitext(os.path.basename(img_path))[0]}.png"
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Live demo of the SNN + GRU concept bottleneck model (Team 07)")
    p.add_argument("--index", type=int, help=f"test photo number 0..{N_TEST - 1}")
    p.add_argument("--image", help="path to any bird photo")
    p.add_argument("--wrong", action="store_true", help="pick a random test photo the model gets wrong")
    p.add_argument("--eval", action="store_true", help="whole test set, all 3 seeds, from the saved features")
    p.add_argument("--seed", type=int, default=0, choices=(0, 1, 2))
    p.add_argument("--fix", type=int, default=8, help="concepts to fix in step 6")
    p.add_argument("--top", type=int, default=8, help="concepts to list in step 4")
    p.add_argument("--no-fig", action="store_true")
    p.add_argument("--open", action="store_true", help="open the saved figure (Windows)")
    args = p.parse_args()
    if args.index is not None and not 0 <= args.index < N_TEST:
        p.error(f"--index must be between 0 and {N_TEST - 1}")
    if args.eval:
        eval_mode(args)
    else:
        demo_mode(args)


if __name__ == "__main__":
    main()
