"""
gonogo_step2.py -- CUB go/no-go, step 2: an ATTENTION READOUT that keeps spatial information.

WHY: every model so far averages its last feature map over space before the decoder (SNN: the 14x14 spike map of
layers.2.6.down.0 at each of the 4 time steps; ResNet-34: the 7x7 layer4 map). Concepts such as "wing colour" or
"bill shape" are local, so the average throws away where they are. Step 1 showed that a concept-loss weight of 3
makes the concepts beat a random layer at NEC = 5 but costs about one point of accuracy; step 2 tries to win that
point back (and more) by letting the model choose where to look.

READOUT (both backbones, same module): multi-head attention pooling with H = 8 heads. Channels are split into 8
groups; head h scores every spatial position with a 1x1 linear map of the full feature vector (C -> 1), a softmax
over positions gives its weights, and it pools its own channel group with them. Output: the same C-dim vector the
mean-pooled models had, so everything after it (GRU per time step for the SNN, MLP for ResNet-34, concept layer,
head) is unchanged. The scoring weights start at ZERO, so at initialisation the readout IS mean pooling (the step-1
model); training can only move away from it. Extra trainable parameters: SNN 12,296 (+0.6%), ResNet-34 4,104
(+0.2%), so the 1.97 M capacity match is kept within 0.5%.

TRAINING: the step-1 lc3 recipe exactly -- joint loss 3 * BCE + CE, the same 8 augmented views per image (same
view parameters, aug_views/cache/view_params.npz), AdamW 1e-3 / 1e-4, batch 32, 50 epochs, cosine LR, grad clip 5,
concept dropout 0.25, epoch selected on held-out class accuracy. Seeds 0-4.

SPATIAL CACHE (built once by --build-cache, ~10.7 GB, resumable): for the 8 train views and the EVAL_TF train_fit /
held_out / test images, the SNN's binary spikes [T=4, 1536, 196] bit-packed (153.6 KB per image) and ResNet-34's
layer4 map [512, 49] float16. Check: the spatial mean of every cached map is compared with the existing pooled
caches on the first 64 images of every split/view (must agree to float16 precision), so the new features are
provably the old ones before pooling.

GO/NO-GO RULE FOR STEP 2 (fixed before running; 5 seeds; paired bootstrap over test images, 95% CI):
  (1) accuracy: SNN + attention (lambda_c 3) minus SNN + mean pooling (lambda_c 3, step 1)  > 0
  (2) concepts still beat random-112 at NEC = 5 (SNN + attention)                          > 0
  (3) SNN + attention within 1 point of ResNet-34 + MLP + attention (SNN - ANN)            > -1
  PASS = (1) and (2) and (3).

STANDALONE: imports existing modules read-only; writes only into spatial_cache/ and gonogo_step2/ (or the _dry
versions); a fingerprint of every other repo file is checked at the end.

USAGE (repo root, project venv python, GPU):
  python gonogo_step2.py --dry-run          # 64 images per split, 1 view, 1 epoch, 5 batches; writes *_dry/ only
  python gonogo_step2.py --build-cache      # ~30-45 min, resumable
  python gonogo_step2.py                    # train 2 models x 5 seeds + NEC + report (resumable)
  python gonogo_step2.py --report
"""
import argparse, json, os, random, sys, time, warnings

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional

import run_aug_views as rav
import run_seeds as rs
import run_seeds_extra as rse
import run_seeds_round3 as r3
import nec_sparse_head as nsh
from models.cbm import SpikingResformerCBM
from anec5_gap_test import build_spiking_backbone
from train_cbm import DEVICE, cosine_lr_schedule

MODELS = ("learned_decoder", "ann34_mlp")
LABEL = {"learned_decoder": "SNN + GRU + attention", "ann34_mlp": "ResNet-34 + MLP + attention"}
SEEDS = (0, 1, 2, 3, 4)
HEADS = 8
LAMBDA_C = 3.0
T, C_SNN, P_SNN, PB = 4, 1536, 196, 25          # PB = bytes per packed 196-bit spatial row
C_ANN, P_ANN = 512, 49
EPOCHS, BATCH, LR, WD = rav.EPOCHS, rav.BATCH_SIZE, rav.LR, rav.WD
NEC_K, N_RANDOM = 5, 112
BOOT_SEED, N_BOOT = 20260826, 10_000
STEP1_RUNS = os.path.join(ROOT, "gonogo_step1", "runs", "lc3")

DRY = False
CACHE = os.path.join(ROOT, "spatial_cache")
OUT = os.path.join(ROOT, "gonogo_step2")


def _guard(path):
    rp = os.path.realpath(path)
    ok = any(rp == os.path.realpath(r) or rp.startswith(os.path.realpath(r) + os.sep) for r in (CACHE, OUT))
    if not ok:
        raise RuntimeError(f"[Guard] refusing to write outside {CACHE} / {OUT}: {path}")
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    return path


def fingerprint():
    skip = [os.path.realpath(p) for p in (CACHE, OUT)]
    fp = {}
    for d, dirs, files in os.walk(ROOT):
        rd = os.path.realpath(d)
        if any(rd == s or rd.startswith(s + os.sep) for s in skip):
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        for f in files:
            try:
                st = os.stat(os.path.join(d, f)); fp[os.path.join(d, f)] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return fp


# ================================================================================================ spatial cache
def cpath(name, bk):
    return os.path.join(CACHE, f"{name}_{bk}.npy")


def cdone(name):
    return os.path.join(CACHE, f"{name}.done.json")


def build_cache(rows_all, limit=None):
    rows_tr = rows_all["train_fit"]
    if not os.path.isfile(rav.PARAMS_PATH):
        raise SystemExit(f"[Stop] {rav.PARAMS_PATH} not found. Run this script inside the folder that holds the "
                         "8-view cache (New folder\\SpikingResformer-cbm), not the git copy.")
    p = rav.load_or_draw_params(rows_tr, write=False)
    jobs = [(f"view{k}", rows_tr, p, k) for k in range(1 if DRY else rav.K_VIEWS)] + \
           [(s, rows_all[s], None, 0) for s in ("train_fit", "held_out", "test")]
    snn = SpikingResformerCBM(backbone=build_spiking_backbone(), n_concepts=112, n_classes=200,
                              readout_type="spike_rate", backbone_dim=C_SNN).to(DEVICE).eval()
    r34 = rse._resnet(34)[0]
    grab = {}
    r34.layer4.register_forward_hook(lambda m, i, o: grab.__setitem__("l4", o))
    for name, rows, pp, k in jobs:
        if os.path.isfile(cdone(name)):
            print(f"[Cache] {name}: done earlier, skipped"); continue
        rows = rows if limit is None else rows[:limit]
        if pp is not None and limit is not None:
            pp = {kk: v[:, :limit] for kk, v in pp.items()}
        n = len(rows)
        ms = np.lib.format.open_memmap(_guard(cpath(name, "snn")), mode="w+", dtype=np.uint8, shape=(n, T, C_SNN, PB))
        ma = np.lib.format.open_memmap(_guard(cpath(name, "r34")), mode="w+", dtype=np.float16, shape=(n, C_ANN, P_ANN))
        loader = torch.utils.data.DataLoader(rav.ViewDataset(rows, pp, k), batch_size=BATCH, shuffle=False, num_workers=0)
        t0 = time.time()
        with torch.no_grad():
            for imgs, idx in loader:
                idx = idx.numpy(); imgs = imgs.to(DEVICE)
                functional.reset_net(snn.backbone)
                snn.backbone(imgs)
                s = snn._hooked_lif._spike_seq                                  # [T, B, C, 14, 14]
                s = s.permute(1, 0, 2, 3, 4).reshape(len(idx), T, C_SNN, P_SNN).to(torch.uint8).cpu().numpy()
                ms[idx[0]:idx[-1] + 1] = np.packbits(s, axis=-1)
                r34(imgs)
                ma[idx[0]:idx[-1] + 1] = grab["l4"].reshape(len(idx), C_ANN, P_ANN).half().cpu().numpy()
        ms.flush(); ma.flush()
        chk = verify(name, ms, ma)
        json.dump({"n": n, "seconds": time.time() - t0, "check": chk}, open(_guard(cdone(name)), "w"), indent=1)
        print(f"[Cache] {name}: {n} images, {time.time() - t0:.0f} s, check {chk}")


def verify(name, ms, ma, m=64):
    """Spatial mean of the cached maps vs the existing pooled caches (first m images)."""
    m = min(m, len(ms))
    snn_mean = unpack(torch.from_numpy(np.asarray(ms[:m]))).mean(-1).numpy()          # [m, T, C]
    ann_mean = np.asarray(ma[:m]).astype(np.float32).mean(-1)                           # [m, 512]
    if name.startswith("view"):
        k = int(name[4:])
        ref_s = np.load(rav.view_path(k, "snn"), mmap_mode="r")[:m].astype(np.float32)
        ref_a = np.load(rav.view_path(k, "r34"), mmap_mode="r")[:m].astype(np.float32)
    else:
        d, key = rav.EVAL_SRC["snn"]; ref_s = np.load(os.path.join(d, f"{name}.npz"))[key][:m].astype(np.float32)
        d, key = rav.EVAL_SRC["r34"]; ref_a = np.load(os.path.join(d, f"{name}.npz"))[key][:m].astype(np.float32)
    es = float(np.abs(snn_mean - ref_s).max()); ea = float(np.abs(ann_mean - ref_a).max() / (np.abs(ref_a).max() + 1e-9))
    ok = es < 2e-3 and ea < 5e-3
    if not ok:
        print(f"[Cache] WARNING {name}: spatial mean differs from the pooled cache (snn max abs {es:.2e}, "
              f"r34 max rel {ea:.2e}) -- check before training")
    return {"snn_max_abs": es, "r34_max_rel": ea, "ok": ok}


_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def unpack(x):
    """uint8 [..., 25] packed bits -> float [..., 196] (works on CPU or GPU tensors)."""
    sh = _SHIFTS.to(x.device)
    bits = (x.unsqueeze(-1) >> sh) & 1
    return bits.reshape(*x.shape[:-1], PB * 8)[..., :P_SNN].float()


# ================================================================================================ model
class AttPool(nn.Module):
    """H-head attention pooling over positions; head h pools channel group h. Zero init = mean pooling."""

    def __init__(self, c, heads=HEADS):
        super().__init__()
        self.c, self.h = c, heads
        self.score = nn.Conv1d(c, heads, 1)
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)

    def forward(self, x):                                   # x [N, C, P]
        a = torch.softmax(self.score(x), dim=-1)            # [N, H, P]
        n, c, p = x.shape
        return (x.view(n, self.h, c // self.h, p) * a.unsqueeze(2)).sum(-1).reshape(n, c)

    def maps(self, x):
        return torch.softmax(self.score(x), dim=-1)


class AttCBM(nn.Module):
    def __init__(self, kind, n_concepts):
        super().__init__()
        self.kind = kind
        self.inner = r3.make_model(kind, n_concepts)
        self.att = AttPool(C_SNN if kind == "learned_decoder" else C_ANN)

    def pooled(self, x):
        if self.kind == "learned_decoder":                  # x [B, T, C, P]
            b = x.shape[0]
            return self.att(x.reshape(b * T, C_SNN, P_SNN)).reshape(b, T, C_SNN)
        return self.att(x)                                  # x [B, 512, 49]

    def features(self, x):
        return self.inner.features(self.pooled(x))

    @property
    def cbl(self):
        return self.inner.cbl

    def forward(self, x, concept_targets=None, concept_dropout_prob=0.0):
        return self.inner(self.pooled(x), concept_targets=concept_targets, concept_dropout_prob=concept_dropout_prob)

    def trainable_parameters(self):
        return list(self.att.parameters()) + list(self.inner.trainable_parameters())


def loss_fn(cs, cl, a, y):
    return LAMBDA_C * F.binary_cross_entropy(cs, a.float()) + F.cross_entropy(cl, y.long())


# ================================================================================================ data
class Feats:
    def __init__(self, kind):
        self.bk = "snn" if kind == "learned_decoder" else "r34"

    def views(self):
        k = 1 if DRY else rav.K_VIEWS
        return [np.load(cpath(f"view{i}", self.bk), mmap_mode="r") for i in range(k)]

    def split(self, s):
        return np.load(cpath(s, self.bk), mmap_mode="r")

    def to_dev(self, arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(DEVICE)
        return unpack(t) if self.bk == "snn" else t.float()


def batches(fe, arr, bs=64):
    for i in range(0, len(arr), bs):
        yield i, fe.to_dev(arr[i:i + bs])


@torch.no_grad()
def scores(model, fe, arr):
    model.eval()
    cs, pr = [], []
    for _, x in batches(fe, arr):
        c, l = model(x)
        cs.append(c.cpu().numpy()); pr.append(l.argmax(1).cpu().numpy())
    return np.concatenate(cs), np.concatenate(pr)


def mean_auc(cs, a):
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(a[:, j], cs[:, j]) for j in range(a.shape[1]) if 0 < a[:, j].mean() < 1]))


# ================================================================================================ training
def run_dir(kind, seed):
    return os.path.join(OUT, "runs", f"{kind}_seed{seed}")


def train(kind, seed, lab, held, test, n_concepts, max_batches=None):
    rd = run_dir(kind, seed)
    if os.path.isfile(os.path.join(rd, "result.json")):
        print(f"[{kind} seed {seed}] done earlier, skipped"); return
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    vgen = torch.Generator().manual_seed(seed + rav.VIEW_DRAW_OFFSET)
    fe = Feats(kind); views = fe.views(); n = len(views[0])
    model = AttCBM(kind, n_concepts).to(DEVICE)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WD)
    at = torch.from_numpy(lab[0][:n]).float().to(DEVICE); yt = torch.from_numpy(lab[1][:n]).long().to(DEVICE)
    best, t0 = {"val_acc": -1.0}, time.time()
    epochs = 1 if DRY else EPOCHS
    for ep in range(1, epochs + 1):
        model.train()
        cosine_lr_schedule(opt, ep - 1, EPOCHS, lr_max=LR)
        v = torch.randint(len(views), (n,), generator=vgen).numpy()
        perm = torch.randperm(n, generator=gen).numpy()
        nb = n // BATCH if max_batches is None else min(max_batches, n // BATCH)
        losses = []
        for bi in range(nb):
            idx = perm[bi * BATCH:(bi + 1) * BATCH]
            xb = fe.to_dev(np.stack([views[v[i]][i] for i in idx]))
            it = torch.from_numpy(idx).to(DEVICE)
            cs, cl = model(xb, concept_targets=at[it], concept_dropout_prob=rav.CONCEPT_DROPOUT)
            loss = loss_fn(cs, cl, at[it], yt[it])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=rav.GRAD_CLIP)
            opt.step(); losses.append(loss.item())
        cs, pr = scores(model, fe, held[0])
        acc = float((pr == held[2]).mean() * 100); auc = mean_auc(cs, held[1])
        if acc > best["val_acc"]:
            best = {"val_acc": acc, "val_auc": auc, "epoch": ep, "state": rs._cpu_state(model)}
        print(f"  {LABEL[kind]} seed {seed} | ep {ep:2d}/{epochs} | loss {np.mean(losses):.3f} | held-out "
              f"{acc:.2f}% / AUC {auc:.4f}{'  *best*' if best['epoch'] == ep else ''} | {(time.time() - t0) / 60:.1f} min")
    model.load_state_dict(best["state"])
    cs, pr = scores(model, fe, test[0])
    res = {"model": kind, "seed": seed, "lambda_c": LAMBDA_C, "heads": HEADS, "selected_epoch": best["epoch"],
           "held_out_acc": best["val_acc"], "test_acc": float((pr == test[2]).mean() * 100),
           "test_concept_auc": mean_auc(cs, test[1]), "concept_err_pct": float(((cs > 0.5) != test[1]).mean() * 100),
           "train_minutes": (time.time() - t0) / 60,
           "extra_params": sum(p.numel() for p in model.att.parameters()),
           "trainable_params": sum(p.numel() for p in model.trainable_parameters())}
    torch.save({"state": best["state"], "result": res}, _guard(os.path.join(rd, "best_acc.pth")))
    np.savez(_guard(os.path.join(rd, "test_outputs_acc.npz")), cs=cs.astype(np.float32), pred=pr)
    json.dump(res, open(_guard(os.path.join(rd, "result.json")), "w"), indent=1)
    print(f"  [Done] {LABEL[kind]} seed {seed}: epoch {best['epoch']} -> test {res['test_acc']:.2f}%, concept AUC "
          f"{res['test_concept_auc']:.4f}, concept errors {res['concept_err_pct']:.2f}%")


# ================================================================================================ NEC = 5
def fast_path():
    def path(X, Y1h, n_lambda=nsh.N_LAMBDA, ratio=nsh.LAMBDA_RATIO, alpha=nsh.ALPHA, max_iter=nsh.MAX_ITER,
             tol=nsh.TOL, log=None):
        Cn = Y1h.shape[1]
        lmax = nsh.lambda_max(X, Y1h, alpha)
        lams = np.exp(np.linspace(np.log(lmax), np.log(lmax / ratio), n_lambda))
        L = nsh.lipschitz(X)
        W = torch.zeros(Cn, X.shape[1], dtype=X.dtype, device=X.device); b = nsh.bias_only(Y1h).to(X.dtype)
        res = []
        for lam in lams:
            W, b, its = nsh.fista(X, Y1h, float(lam), W, b, L, alpha, max_iter, tol)
            nec = float((W != 0).sum()) / Cn
            res.append((float(lam), W.clone(), b.clone(), nec, its))
            if nec > NEC_K + 1:
                break
        return res
    nsh.regularization_path = path


@torch.no_grad()
def reps(model, fe, arr):
    model.eval(); H, Z = [], []
    for _, x in batches(fe, arr):
        h = model.features(x); H.append(h.cpu().numpy()); Z.append(model.cbl.linear(h).cpu().numpy())
    return np.concatenate(H), np.concatenate(Z)


def nec_unit(kind, seed, splits, n_concepts):
    p = os.path.join(OUT, "nec", f"{kind}_seed{seed}.json")
    if os.path.isfile(p):
        return json.load(open(p, encoding="utf-8"))
    ck = torch.load(os.path.join(run_dir(kind, seed), "best_acc.pth"), map_location="cpu", weights_only=False)
    model = AttCBM(kind, n_concepts).to(DEVICE); model.load_state_dict(ck["state"])
    fe = Feats(kind)
    H, Z, y = {}, {}, {}
    for s in ("train_fit", "held_out", "test"):
        H[s], Z[s] = reps(model, fe, fe.split(s)); y[s] = splits[s][2].astype(np.int64)[:len(H[s])]
    g = torch.Generator().manual_seed(nsh.RANDOM_SEED_BASE + 1000 * seed + sum(map(ord, kind)))
    Wr = torch.randn(N_RANDOM, H["train_fit"].shape[1], generator=g, dtype=torch.float64).numpy()
    out = {"model": kind, "seed": seed}
    for name, X in (("trained", Z), ("random112", {s: H[s].astype(np.float64) @ Wr.T for s in H})):
        r = nsh.nec_evaluate(X["train_fit"], y["train_fit"], {"held_out": (X["held_out"], y["held_out"]),
                                                              "test": (X["test"], y["test"])},
                             200, targets=(NEC_K,), device=DEVICE, dtype=torch.float32)
        out[name] = {"test_acc": r["targets"][NEC_K]["test_acc"],
                     "test_correct": (r["preds"][NEC_K]["test"] == y["test"]).tolist()}
        print(f"   NEC=5 {LABEL[kind]} seed {seed} {name}: test {out[name]['test_acc']:.2f}%")
    json.dump(out, open(_guard(p), "w"))
    return out


# ================================================================================================ report
def boot(ca, cb):
    d = (np.asarray(ca, float) - np.asarray(cb, float)).mean(0) * 100
    rng = np.random.default_rng(BOOT_SEED); n = len(d)
    m = np.concatenate([d[rng.integers(0, n, size=(1000, n))].mean(1) for _ in range(N_BOOT // 1000)])
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def report(seeds, cids):
    corr = lambda path: np.load(path)["pred"] == cids
    att = {k: [corr(os.path.join(run_dir(k, s), "test_outputs_acc.npz")) for s in seeds] for k in MODELS}
    mean = {k: [corr(os.path.join(STEP1_RUNS, f"{k}_seed{s}", "test_outputs_acc.npz")) for s in seeds] for k in MODELS}
    nec = {k: [json.load(open(os.path.join(OUT, "nec", f"{k}_seed{s}.json"))) for s in seeds] for k in MODELS}
    acc = lambda L: float(np.mean([c.mean() * 100 for c in L]))
    g1 = boot(att["learned_decoder"], mean["learned_decoder"])
    g2 = boot([u["trained"]["test_correct"] for u in nec["learned_decoder"]],
              [u["random112"]["test_correct"] for u in nec["learned_decoder"]])
    g3 = boot(att["learned_decoder"], att["ann34_mlp"])
    ga = boot(att["ann34_mlp"], mean["ann34_mlp"])
    gA = boot([u["trained"]["test_correct"] for u in nec["ann34_mlp"]], [u["random112"]["test_correct"] for u in nec["ann34_mlp"]])
    p1, p2, p3 = g1[1] > 0, g2[1] > 0, g3[1] > -1
    J = {"seeds": list(seeds), "snn_att_acc": acc(att["learned_decoder"]), "snn_mean_acc": acc(mean["learned_decoder"]),
         "ann_att_acc": acc(att["ann34_mlp"]), "ann_mean_acc": acc(mean["ann34_mlp"]),
         "gain_snn_att_minus_mean": g1, "snn_concepts_minus_random112": g2, "snn_minus_ann_att": g3,
         "gain_ann_att_minus_mean": ga, "ann_concepts_minus_random112": gA, "pass": [p1, p2, p3], "PASS": p1 and p2 and p3}
    f = lambda g: f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]"
    md = [f"# Go/no-go step 2: attention readout ({len(seeds)} seeds, lambda_c = {LAMBDA_C:g})", "",
          "| Model | Mean pooling (step 1) | Attention | Gain (95% CI) | NEC-5 concepts - random-112 (attention) |",
          "|---|---|---|---|---|",
          f"| SNN + GRU | {J['snn_mean_acc']:.2f} | {J['snn_att_acc']:.2f} | {f(g1)} | {f(g2)} |",
          f"| ResNet-34 + MLP | {J['ann_mean_acc']:.2f} | {J['ann_att_acc']:.2f} | {f(ga)} | {f(gA)} |", "",
          f"SNN - ANN with attention: {f(g3)}", "",
          "Rule (fixed before running): (1) SNN attention gain CI > 0; (2) SNN concepts beat random-112 (CI > 0); "
          "(3) SNN - ANN (attention) CI > -1.", "",
          f"Verdict: (1) {'pass' if p1 else 'fail'}, (2) {'pass' if p2 else 'fail'}, (3) {'pass' if p3 else 'fail'} -> "
          f"**{'PASS' if J['PASS'] else 'FAIL'}**"]
    open(_guard(os.path.join(OUT, "report.md")), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(J, open(_guard(os.path.join(OUT, "report.json")), "w"), indent=1)
    print("\n".join(md))


# ================================================================================================ main
def main(a):
    global DRY, CACHE, OUT
    DRY = a.dry_run
    if DRY:
        CACHE, OUT = CACHE + "_dry", OUT + "_dry"
    os.makedirs(OUT, exist_ok=True)
    before = fingerprint()
    rows_all = rav.split_rows()
    n_concepts = rav.n_concepts_of(rows_all["train_fit"])
    limit = 64 if DRY else None
    if a.build_cache or DRY:
        build_cache(rows_all, limit)
        if not DRY:
            return
    lab = rav.train_labels(rows_all["train_fit"])
    splits = {}
    for s in ("train_fit", "held_out", "test"):
        z = np.load(os.path.join(rs.CACHE_DIR, f"{s}.npz"))
        splits[s] = (z["attrs"], z["cids"]) if s == "train_fit" else (z["attrs"], z["cids"])
        splits[s] = (None, z["attrs"], z["cids"])
    fast_path()
    seeds = (0,) if DRY else SEEDS
    if not a.report:
        for s in seeds:
            for k in MODELS:
                fe = Feats(k)
                held = (fe.split("held_out"), splits["held_out"][1][:limit], splits["held_out"][2][:limit])
                test = (fe.split("test"), splits["test"][1][:limit], splits["test"][2][:limit])
                train(k, s, lab, held, test, n_concepts, max_batches=1 if DRY else None)
        for s in seeds:
            for k in MODELS:
                nec_unit(k, s, splits, n_concepts)
    if DRY:
        print("[Dry run] cache build + check, attention model training, test scoring and NEC = 5 all ran; outputs only in "
              "spatial_cache_dry/ and gonogo_step2_dry/ (safe to delete).")
        return
    report(seeds, splits["test"][2])
    after = fingerprint()
    changed = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    print(f"\n[Guard] {'WARNING changed: ' + str(changed[:5]) if changed else 'PROTECTED FILES UNCHANGED.'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--report", action="store_true")
    main(ap.parse_args())
