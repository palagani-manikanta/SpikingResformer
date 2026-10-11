"""Figures for the paper. Every number is copied from a report in the repository (source noted per block);
the reliability diagram is recomputed from the saved seed-0 test outputs + saved Platt parameters."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "Liberation Serif", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#3a3936", "ytick.color": "#3a3936",
    "axes.labelcolor": "#1d1d1b", "text.color": "#1d1d1b", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})

SNN, R34, R18, R50, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#6b6a66"
COL = 3.45


def style(ax):
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Fig. 2: accuracy vs compute energy
# accuracy: aug_views/results/aug_views_report.md (ClassAcc-selected, mean +- std, 3 seeds)
# energy:   energy_audit_v2/energy_audit_v2_report.md (45 nm op counts, stem once, mJ per image)
pts = [  # label, energy, acc, std, color, marker, filled, label offset (pts)
    ("SNN + GRU (ours)", 2.978, 58.60, 0.20, SNN, "o", True, (6, -9)),
    ("ResNet-18, linear", 8.343, 56.92, 0.51, R18, "^", False, (6, 3)),
    ("ResNet-18 + MLP", 8.351, 56.24, 0.15, R18, "^", True, (6, -8)),
    ("ResNet-34, linear", 16.851, 59.07, 0.68, R34, "s", False, (-62, 5)),
    ("ResNet-34 + MLP", 16.860, 58.69, 0.78, R34, "s", True, (-58, -10)),
    ("ResNet-50, linear", 18.802, 62.50, 0.33, R50, "D", False, (-62, -2)),
]
fig, ax = plt.subplots(figsize=(COL, 2.05))
for lab, e, a, s, c, m, filled, off in pts:
    ax.errorbar(e, a, yerr=s, fmt=m, ms=5.2, color=c, mfc=c if filled else "white", mec=c, mew=1.1,
                elinewidth=0.9, capsize=2, zorder=3)
    ax.annotate(lab, (e, a), xytext=off, textcoords="offset points", fontsize=6.9, color="#1d1d1b")
ax.set_xlabel("Estimated compute energy per image (mJ, 45 nm)")
ax.set_ylabel("CUB test accuracy (%)")
ax.set_xlim(0, 21)
ax.set_ylim(55, 63.5)
style(ax)
ax.annotate("", xy=(2.978, 60.4), xytext=(16.86, 60.4),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
ax.text(9.9, 60.6, "5.7× less compute energy, no significant accuracy gap", ha="center", fontsize=6.8, color=MUTED)
fig.savefig(os.path.join(OUT, "acc_energy.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: calibration, intervention, NEC
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0), gridspec_kw={"wspace": 0.36})

# (a) reliability diagram, SNN + GRU seed 0, pooled over 112 concepts x 5,794 test images
base = sys.argv[1] if len(sys.argv) > 1 else None
if base:
    cs = np.load(os.path.join(base, "aug_views/runs/learned_decoder_seed0/test_outputs_acc.npz"))["cs"].astype(np.float64)
    attrs = np.load(os.path.join(base, "seeds/cache/test.npz"))["attrs"].astype(np.float64)
    pl = json.load(open(os.path.join(base, "calibration_all/units/learned_decoder_seed0_acc.json")))["platt"]
    z = np.log(np.clip(cs, 1e-7, 1 - 1e-7) / (1 - np.clip(cs, 1e-7, 1 - 1e-7)))
    cal = 1 / (1 + np.exp(-(np.array(pl["a"])[None] * z + np.array(pl["b"])[None])))
    edges = np.linspace(0, 1, 16)

    def rel(p, y):
        idx = np.clip(np.digitize(p.ravel(), edges) - 1, 0, 14)
        conf = np.array([p.ravel()[idx == k].mean() if (idx == k).any() else np.nan for k in range(15)])
        acc = np.array([y.ravel()[idx == k].mean() if (idx == k).any() else np.nan for k in range(15)])
        return conf, acc
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color="#bdbcb7", lw=0.8, zorder=1)
    c0, a0 = rel(cs, attrs)
    c1, a1 = rel(cal, attrs)
    ax.plot(c0, a0, "--", marker="o", ms=2.8, color=SNN, lw=1.1, mfc="white", label="raw (ECE 0.076)")
    ax.plot(c1, a1, "-", marker="o", ms=2.8, color=SNN, lw=1.3, label="calibrated (ECE 0.017)")
    ax.set_xlabel("Predicted concept probability")
    ax.set_ylabel("Fraction of concepts present")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", handlelength=1.6, fontsize=6.4, borderaxespad=0.2)
    style(ax)
    ax.set_title("(a) Concept calibration, SNN + GRU", loc="left")

# (b) intervention: calibration_all/calibration_all_report.md Part B (head retrained per arm, mean of 3 seeds)
fr = np.array([0, 10, 25, 50, 75, 100])
iv = {"SNN raw": ([59.04, 66.02, 75.37, 91.85, 97.20, 96.92], SNN, "--"),
      "SNN calibrated": ([58.92, 69.22, 81.69, 95.73, 98.49, 98.46], SNN, "-"),
      "ResNet-34 raw": ([58.68, 64.48, 72.74, 91.02, 97.29, 97.15], R34, "--"),
      "ResNet-34 calibrated": ([58.53, 66.89, 78.53, 94.94, 98.25, 98.29], R34, "-")}
ax = axs[1]
for lab, (v, c, ls) in iv.items():
    ax.plot(fr, v, ls, color=c, lw=1.2, marker="o", ms=2.6, mfc="white" if ls == "--" else c, label=lab)
ax.set_xlabel("Concepts replaced by true values (%)")
ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(fr)
ax.set_ylim(55, 100)
ax.legend(loc="lower right", handlelength=1.8, bbox_to_anchor=(1.03, -0.02), fontsize=6.4)
style(ax)
ax.set_title("(b) Simulated concept intervention", loc="left")

# (c) NEC: nec_sparse/nec_sparse_report.json per_target, mean over 3 seeds
nec = [5, 10, 15, 20, 25, 30]
nv = {"SNN": ([42.52, 56.82, 58.54, 58.54, 58.53, 58.53], SNN, "-"),
      "SNN, random CBL": ([44.25, 56.99, 58.26, 58.26, 58.25, 59.19], SNN, ":"),
      "ResNet-34": ([44.82, 55.66, 56.36, 56.73, 56.73, 56.73], R34, "-"),
      "ResNet-34, random CBL": ([48.56, 56.78, 56.92, 56.92, 57.60, 57.62], R34, ":"),
      "ResNet-18": ([41.43, 54.15, 54.79, 55.20, 55.25, 55.11], R18, "-")}
ax = axs[2]
for lab, (v, c, ls) in nv.items():
    ax.plot(nec, v, ls, color=c, lw=1.2, marker="o", ms=2.6, mfc="white" if ls == ":" else c, label=lab)
ax.set_xlabel("Number of effective concepts per class (NEC)")
ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(nec)
ax.set_ylim(40, 61)
ax.legend(loc="lower right", handlelength=1.8, bbox_to_anchor=(1.03, -0.02), fontsize=6.4)
style(ax)
ax.set_title("(c) Sparse final layer (VLG-CBM protocol)", loc="left")
fig.savefig(os.path.join(OUT, "calib_interv_nec.pdf"))
plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: energy vs memory cost per byte
# energy_memory/energy_memory_report.md, 8-bit reference scenario: compute mJ + bytes per image x pJ/B
models = [("SNN + GRU (ours)", 2.978, 282.25, SNN, "-"),
          ("ResNet-34 + MLP", 16.860, 30.44, R34, "-"),
          ("ResNet-18 + MLP", 8.351, 17.83, R18, "-")]
e = np.linspace(0, 340, 400)
fig, ax = plt.subplots(figsize=(COL, 2.0))
ax.axvspan(162.5, 325, color="#f1f0ec", zorder=0)
ax.text(243.75, 101, "off-chip DRAM\n(162.5–325 pJ/B)", ha="center", va="top", fontsize=6.6, color=MUTED)
ax.axvline(12.5, color="#bdbcb7", lw=0.8, zorder=1)
ax.text(15, 101, "1 MB SRAM\n(12.5 pJ/B)", ha="left", va="top", fontsize=6.6, color=MUTED)
for lab, comp, mb, c, ls in models:
    ax.plot(e, comp + mb * 1e6 * e * 1e-12 * 1e3, ls, color=c, lw=1.4, label=lab)
be = (16.860 - 2.978) / ((282.25 - 30.44) * 1e-3)
ax.plot([be], [2.978 + 282.25e-3 * be], "o", ms=4, color="#1d1d1b", zorder=4)
ax.annotate(f"break-even vs ResNet-34 + MLP\n{be:.1f} pJ/B", (be, 2.978 + 282.25e-3 * be), xytext=(18, 63),
            textcoords="data", fontsize=6.6, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax.set_xlim(0, 340)
ax.set_ylim(0, 102)
ax.set_xlabel("Memory access energy (pJ per byte)")
ax.set_ylabel("Energy per image (mJ)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.2, handlelength=1.8)
style(ax)
fig.savefig(os.path.join(OUT, "energy_memory.pdf"))
plt.close(fig)
print("break-even", round(be, 2))
print("ok")
