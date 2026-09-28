# Review fixes: corrected results (September 2026)

This file is the **current source of truth** for the project's claims. It replaces the headline numbers in `README.md` and `PROJECT_SUMMARY.md`, which are kept unchanged for history. Every number below comes from a report file in this repository (listed at the end). No original script, checkpoint or result file was modified by the review work.

> **Pending:** every capacity-matched comparison below uses the fast cached recipe (no augmentation). The augmented-recipe comparison is still backed by **seed 0 only** (GRU 59.48% vs fair ANN 57.23%). Augmented seeds 1–2 for the GRU, the shuffled-timestep GRU and the capacity-matched ANNs are needed before any accuracy claim; this file will be updated with the result.

---

## 1. What the evidence supports (as of commit `78c2cb9`, 28 Sep 2026)

This section replaces the earlier Section 1 and the "Update after review round 2–3" note that sat above it. The earlier version compared the GRU model (1.97M trainable parameters) with an ANN that had **no decoder** (80k parameters); with equal-size decoders three of its claims no longer hold. They are recorded in Section 2.

Unless stated otherwise: test split n = 5,794; seeds 0–2; fast cached recipe (frozen-backbone features, no augmentation); checkpoint chosen by held-out class accuracy; gaps are pooled over seeds with 95% paired-bootstrap CIs (10,000 resamples). Gap = GRU minus the other model; for ECE, negative means the GRU is better calibrated.

### 1a. Claims

| Claim | Status | Evidence |
|:---|:---|:---|
| Correcting concepts (intervention) improves accuracy monotonically | **Holds (3 seeds)** | 0 monotonicity violations; about 60% → 98.3–98.5% at 100% intervention (Section 5) |
| Calibrated concepts in the decision path are accuracy-neutral and make intervention more effective | **Holds (learned_decoder, 3 seeds)** | −0.07 pts [−0.55, +0.39]; accuracy at 25% intervention 76.63% → 81.64% (Section 5). For `pre_reset_vmem` it costs −1.29 pts [−1.96, −0.63] |
| Keeping per-timestep spike responses beats averaging them (concept quality) | **Holds (3 seeds)** | GRU vs equal-size MLP on T-averaged spikes: AUC +0.0317 [+0.0302, +0.0331], ECE −0.0338 [−0.0347, −0.0327]. Accuracy: +0.05 [−0.83, +0.94], no difference |
| A shared recurrent decoder beats an equal-size MLP on concatenated timesteps | **Holds (3 seeds)** | Acc +3.84 [+3.03, +4.68], AUC +0.0151 [+0.0138, +0.0163], ECE −0.0393 [−0.0402, −0.0382] (`seeds_extra/`) |
| Extra decoder parameters, not time, drive class accuracy | **Holds (3 seeds)** | MLP-no-time vs spike_rate: +3.13 pts [+1.95, +4.29]; GRU vs MLP-no-time: +0.05 pts (no difference) |
| SNN concepts are better than ANN concepts | **Does not hold** against equal-size decoders | GRU vs ResNet-18 + MLP: AUC −0.0113 [−0.0137, −0.0090]; vs ResNet-34 + MLP: AUC −0.0185 [−0.0208, −0.0163] (ANN better on all 3 seeds). vs ResNet-50 + MLP: +0.0027 [+0.0005, +0.0048], average only, and that decoder is a narrow 2048→418→2048 bottleneck. The GRU does beat ANNs **without** a decoder (vs ResNet-18 linear +0.0505, ResNet-34 linear +0.0445, ResNet-50 linear +0.0278), which is a capacity difference, not a spiking advantage |
| SNN concepts are better calibrated than ANN concepts | **Does not hold** consistently | vs ResNet-18 + MLP: ECE −0.0033 [−0.0044, −0.0021], average only (seeds disagree in sign); vs ResNet-34 + MLP: +0.0036 [+0.0024, +0.0049] (ANN better) |
| The order of the 4 timesteps carries information | **Does not hold** | A GRU trained on randomly shuffled timesteps is better than the normal GRU: GRU − shuffled acc −1.89 [−2.61, −1.16], AUC −0.0036 [−0.0044, −0.0027], ECE +0.0019 [+0.0012, +0.0025] (`seeds_round3/`). The −15.36-pt drop when reversing the order at test time (3a) shows the normal GRU *uses* order, not that order is *needed*. Expected on static images, where every timestep sees the same input |
| GRU class accuracy matches or exceeds the ANN | **Not established** | Augmented, seed 0 only: 59.48% vs 57.23% (ResNet-18 linear), +2.24 [+0.88, +3.57]. Cached, 3 seeds: the GRU is below every ANN (vs ResNet-18 + MLP −4.54 [−5.67, −3.43], vs ResNet-34 + MLP −7.23 [−8.38, −6.08]). The cached recipe penalises decoders: adding an MLP lowers ANN accuracy by 2.8–11.5 pts (ResNet-18 56.18 → 52.57, ResNet-34 58.05 → 55.25, ResNet-50 60.98 → 49.53), so it cannot settle this |
| The SNN uses less energy than the ANNs (45 nm op-count estimate) | **Holds, as an estimate** | 2.978 mJ per image with the stem computed once. 6.05× vs the same architecture run densely; 5.66× vs ResNet-34 + MLP; 2.80× vs ResNet-18 (± MLP); 6.32× vs ResNet-50 + MLP (`energy_audit_v2/`). The stem-once wrapper in `energy_audit_v2.py` gives identical predictions; the default model path still recomputes the stem every timestep (3.91× same-architecture). Memory access is not counted |

### 1b. Capacity-matched comparison (cached recipe, mean ± std over seeds 0–2)

| Model | Trainable params | Test acc (%) | Concept AUC | Concept ECE | Energy / image |
|:---|:---:|:---:|:---:|:---:|:---:|
| SNN + GRU (learned_decoder) | 1,967,288 | 48.03 ± 0.54 | 0.8980 | 0.0961 | 2.978 mJ |
| SNN + GRU, shuffled-timestep training | 1,967,288 | 49.92 ± 0.93 | 0.9016 | 0.0943 | 2.978 mJ |
| SNN + MLP on T-averaged spikes | 1,966,328 | 47.98 ± 0.32 | 0.8664 | 0.1299 | — |
| SNN + MLP on concatenated timesteps | 1,970,591 | 44.18 ± 0.66 | 0.8829 | 0.1355 | — |
| SNN spike_rate (no decoder) | 194,744 | 44.85 ± 0.34 | 0.8653 | 0.1217 | — |
| ResNet-18, linear (fair ANN) | 80,056 | 56.18 ± 0.63 | 0.8475 | 0.1213 | 8.343 mJ |
| ResNet-18 + MLP | 1,967,593 | 52.57 ± 0.50 | 0.9093 | 0.0995 | 8.351 mJ |
| ResNet-34, linear | 80,056 | 58.05 ± 0.38 | 0.8535 | 0.1198 | 16.851 mJ |
| ResNet-34 + MLP | 1,967,593 | 55.25 ± 0.13 | 0.9165 | 0.0925 | 16.860 mJ |
| ResNet-50, linear | 252,088 | 60.98 ± 0.50 | 0.8702 | 0.1059 | 18.802 mJ |
| ResNet-50 + MLP (2048→418→2048) | 1,966,682 | 49.53 ± 1.03 | 0.8953 | 0.1007 | 18.810 mJ |

ImageNet-1K top-1 of the frozen backbones: SpikingResformer-Ti 74.38%, ResNet-18 69.76%, ResNet-34 73.31%, ResNet-50 76.13%.

### 1c. Caveats that apply to every row

- **No augmentation** in the cached recipe; decoder models overfit in it (GRU train loss ≈ 0.15). Accuracy comparisons from this recipe favour linear heads.
- **Concept metrics are read at epochs chosen for class accuracy.** Concept AUC and class accuracy peak at different epochs (the original ANN baseline peaked in AUC at epoch 2 and in accuracy at epoch 28); AUC-selected numbers have not been reported yet.
- **Concept labels are class-level** (Koh et al.'s 112 attributes, majority vote on the train split; `process_attributes.py`), so the concepts come close to encoding the class. This is why 100% intervention reaches about 98%.
- **Energy is a 45 nm operation-count estimate**: memory access, membrane-state updates and ANN zero-skipping are not counted. Only a hardware measurement settles the ratio.
- Three seeds give a rough view of training variance; bootstrap CIs resample test images only.

**One-line summary:** a calibrated, correctable ante-hoc CBM on a frozen spiking backbone reaches concept quality close to ANN backbones with equal-size decoders (0.011–0.019 lower concept AUC than ResNet-18/34 + MLP) at 2.8–5.7× lower estimated energy; keeping per-timestep spikes helps concept quality, but timestep order does not, and accuracy parity with the ANN is not established.

---

## 2. Corrections to earlier claims

| Earlier claim (README / PROJECT_SUMMARY) | Corrected | Why |
|:---|:---|:---|
| ANN baseline 58.82%; GRU "beats baseline" (+0.66 pts) | Fair ANN **57.23%**; GRU +2.24 pts, **one seed** | Old ANN trained on the full train split and selected its checkpoint on the test split. The fair ANN uses the same 5,095 / 899 split, held-out selection and recipe as the SNN (`ann_baseline_fair/`) |
| Energy **7.27×** | **6.05×** (learned_decoder), **6.10×** (pre_reset_vmem), stem once, truncated at the tapped layer; **3.91× / 3.93×** with the stem recomputed every time step, as the code currently runs | The original counter under-counted conv MACs about 20× (182.9M vs 3.74G per image per step) and used one network-wide spike rate instead of per-layer rates (`energy_audit/`) |
| pre_reset_vmem "1.15× more efficient" (README) | 6.10× / 3.93× | Same backbone as above; readout cost is negligible |
| PROJECT_SUMMARY: learned_decoder 58.78% vs ANN 58.82% (+0.03) | Current checkpoint: **59.48%** vs fair ANN 57.23% | PROJECT_SUMMARY was written before the final checkpoint and before the fair baseline |
| "Time is why the SNN works" (implied) | Time explains about **21%** of the GRU's +14-pt gain over spike_rate; **79%** comes from decoder parameters | 3c: equal-parameter MLP without time reaches 56.51% vs GRU 59.48% and spike_rate 45.44% |
| ICRC "collapses at 75% / 100%" | In the calibration ablation (retrained heads, 3 seeds) there are **0** monotonicity violations and 98.3–98.5% accuracy at 100% intervention | The collapse result for the shipped head was not re-run in this review |
| Earlier Section 1 (27–28 Sep): temporal GRU decoding gives more accurate and better-calibrated concepts than the ANN (AUC +0.0505, ECE −0.0252) | Holds only against an ANN **without** a decoder; against equal-size decoders ResNet-18/34 + MLP have higher concept AUC (−0.0113 / −0.0185 for the GRU) | The fair ANN had 80k trainable parameters vs 1.97M for the GRU model (`seeds_extra/`) |
| Earlier Section 1: time (not extra parameters) improves concept quality | Per-timestep information helps (GRU vs T-averaged MLP, AUC +0.0317), but the order of the timesteps does not | A GRU trained on shuffled timesteps is as good or better (`seeds_round3/`) |
| Earlier Section 1: the GRU decoder depends on spike order | The normal GRU *uses* order (−15.36 pts when reversed), but order is not *needed* | Reversed or permuted input is out-of-distribution for a GRU trained only on natural order; the shuffled-training control removes that confound |

Not re-checked in this review: the Cohen's d values in `gate_result.json` / PROJECT_SUMMARY Criterion A. Treat them with caution.

---

## 3. What the decoder does (steps 3a–3c, seed 0, augmented recipe)

| Model | Decoder params | Uses time order | Test acc | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|:---:|:---:|
| spike_rate (no decoder) | 0 | No | 45.44% | 0.865 | 0.119 |
| MLP without time | 1,771,584 | No | 56.51% | 0.878 | 0.107 |
| GRU learned_decoder | 1,772,544 | Yes | 59.48% | 0.919 | 0.079 |
| Fair ANN (ResNet-18) | — | — | 57.23% | 0.852 | 0.116 |

- GRU − MLP: +2.97 pts [+1.90, +4.04], McNemar p = 8.5e−8.
- MLP − spike_rate: +11.06 pts [+9.80, +12.34].
- Timing shuffle (3a, no retraining): reversed −15.36 pts, averaged −4.59 pts, random orders −0.33 to −13.43 pts. The GRU was trained on natural order only, so reordered input is also out-of-distribution.

---

## 4. Multi-seed check (step 4)

Seeds 0, 1, 2. Both backbones are frozen, so features were cached once; **these runs train without augmentation** and are compared only with each other. The cache reproduces all four seed-0 checkpoints exactly (differences < 0.01 pts).

| Model | Test acc (mean ± std) | Concept AUC | Concept ECE |
|:---|:---:|:---:|:---:|
| GRU learned_decoder | 48.03 ± 0.54 | 0.8980 ± 0.0025 | 0.0961 ± 0.0106 |
| MLP without time | 47.98 ± 0.32 | 0.8664 ± 0.0004 | 0.1299 ± 0.0073 |
| spike_rate | 44.85 ± 0.34 | 0.8653 ± 0.0002 | 0.1217 ± 0.0004 |
| Fair ANN | 56.18 (seeds 56.73 / 56.32 / 55.49) | ~0.848 | ~0.121 |

| Comparison | Accuracy | Concept AUC | Concept ECE |
|:---|:---|:---|:---|
| GRU vs fair ANN | −8.15 (ANN better) | **+0.0505, all 3 seeds** | **−0.0252, all 3 seeds** |
| GRU vs MLP without time | +0.05 (no difference) | **+0.0317, all 3 seeds** | **−0.0338, all 3 seeds** |
| MLP vs spike_rate | **+3.13, all 3 seeds** | +0.0011 (no difference) | worse |

Without augmentation the decoder models overfit (GRU train loss ≈ 0.15) while the ANN is barely affected (57.2% → 56.2%). This recipe therefore cannot confirm or reject the augmented accuracy result; it does confirm the concept-quality advantage.

---

## 5. Calibration ablation (system with vs. without the calibration novelty)

Head retrained on raw vs. per-concept-Platt-calibrated concepts, 3 seeds each.

| Readout | Accuracy change | Accuracy at 25% intervention (without → with) |
|:---|:---|:---|
| learned_decoder | −0.07 pts [−0.55, +0.39] (no significant change) | 76.63% → 81.64% |
| pre_reset_vmem | −1.29 pts [−1.96, −0.63] | 76.99% → 83.64% |

Feeding calibrated concepts into the already-trained head **without** retraining costs −1.73 pts (learned_decoder) and −4.42 pts (pre_reset_vmem), which is why the shipped model keeps calibration display-only.

---

## 6. Where each number comes from

| Topic | Script | Report |
|:---|:---|:---|
| Calibration ablation | `ablation_calibrated_head.py` | `ablation_calibration/results/` |
| Energy (step 1) | `energy_audit.py` | `energy_audit/` |
| Fair ANN (step 2) | `ann_baseline_fair.py` | `ann_baseline_fair/results/` |
| Timing shuffle (3a) | `timing_shuffle_test.py` | `timing_shuffle/results/` |
| spike_rate fair (3b) | `train_spike_rate_fair.py` | `spike_rate_fair/results/` |
| MLP without time (3c) | `train_mlp_notime.py` | `mlp_notime/results/` |
| Multi-seed (step 4) | `run_seeds.py` | `seeds/results/seeds_report.md` |
| Capacity-matched ANNs, concatenated-timestep MLP | `run_seeds_extra.py` | `seeds_extra/results/seeds_extra_report.md` |
| Shuffled-timestep GRU, ResNet-50 | `run_seeds_round3.py` | `seeds_round3/results/seeds_round3_report.md` |
| Energy vs all ANN baselines, stem-once verification | `energy_audit_v2.py` | `energy_audit_v2/energy_audit_v2_report.md` |
| Concept labels (Koh 112, class-level majority vote) | `process_attributes.py` | `processed_attributes.csv` (git-ignored, with the dataset) |
