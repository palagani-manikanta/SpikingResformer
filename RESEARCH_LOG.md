# Research Log — CBM + SpikingResformer

## Entry: 2026-08-26 — Major Fixes Applied

### Problem Statement
The CBM (Concept Bottleneck Model) built on top of SpikingResformer had three critical issues:
1. **Intervention collapse**: Oracle accuracy (100% ground-truth concepts) dropped BELOW baseline (0% intervention), violating monotonicity. The classification head couldn't handle clean concept inputs because it was trained only on noisy CBL predictions.
2. **Calibration leakage**: Temperature/Platt scaling was fit on data the CBL/head had already seen during training, understating calibration benefits.
3. **ANN-SNN accuracy gap**: The learned_decoder readout (58.47%) nearly matched ANN (58.82%), but training was limited to 30 epochs with no warmup.

### Changes Made

#### 1. Concept Dropout (Fixes Intervention Collapse)
**File**: `models/cbm.py`, `train_cbm.py`, `ann_baseline_cbm.py`

**Mechanism**: During training, randomly replace a fraction of CBL-predicted concept scores with ground-truth {0,1} values before feeding to the classification head. This teaches the head to handle BOTH noisy predictions AND clean ground-truth inputs.

**Implementation**:
- `SpikingResformerCBM.forward()` now accepts `concept_targets` and `concept_dropout_rate` parameters
- When `training=True` and `concept_targets` is provided, per-concept Bernoulli masking replaces CBL outputs with ground-truth
- At eval time (including intervention testing), no dropout is applied — the head sees whatever it's given (predicted or intervened)
- CLI: `--concept-dropout 0.3` (default 0.0, recommended 0.3 for intervention experiments)

**Expected effect**: The head learns that concepts can be either noisy (CBL-predicted) or clean (ground-truth), so intervention with perfect concepts no longer causes distribution shift.

#### 2. Proper 4-Way Split (Fixes Calibration Leakage)
**File**: `train_cbm.py`

**Mechanism**: Carve a calibration split from the original train set BEFORE training, so calibration data is never seen by the CBL/head.

**Implementation**:
- Split: 85% train / 15% calibration (configurable via `--calib-fraction`)
- Saved to `evaluation_results/train_calib_split.json` with image paths for reproducibility
- Calibration scripts (`calibration_ece.py`, `calibration_platt.py`) should read this split instead of carving from train post-hoc
- CLI: `--calib-fraction 0.15` (default 0.15)

**Note**: Existing checkpoints were trained on the FULL train split (no calibration holdout). Retraining with the 4-way split is needed for paper-final calibration numbers.

#### 3. Training Improvements (Closes ANN-SNN Gap)
**File**: `train_cbm.py`, `ann_baseline_cbm.py`

**Changes**:
- Default epochs increased from 30 to 50 (more convergence time)
- Added 5-epoch linear warmup before cosine annealing (more stable early training)
- ANN baseline updated to match (50 epochs, warmup, concept_dropout=0.3)

**New CLI args**:
- `--warmup-epochs 5` (default 5)
- `--epochs 50` (default 50, was 30)

#### 4. Path Portability (Codebase Cleanup)
**File**: `train_cbm.py`

**Before**: Hardcoded Windows paths (`C:\Users\palag\...`)
**After**: Environment variables with sensible defaults:
- `SPIKING_RESFORMER_CKPT` → checkpoint path
- `CUB_DATA_DIR` → CUB dataset path
- Falls back to relative paths under repo root

### How to Retrain (Recommended Commands)

```bash
# Spiking CBM with concept dropout + 4-way split
python train_cbm.py --readout learned_decoder --concept-dropout 0.3 --epochs 50

# ANN baseline with matching config
python ann_baseline_cbm.py

# Evaluate intervention consistency after retraining
python intervention_consistency.py --readout learned_decoder

# Calibrate on the held-out calibration split
python calibration_platt.py --readout learned_decoder
```

### Expected Outcomes After Retraining
1. **Intervention monotonicity**: Oracle accuracy should be >= baseline accuracy (no collapse)
2. **Calibration**: Platt scaling ECE should improve more than before (no leakage)
3. **Accuracy**: 50 epochs + warmup should push learned_decoder closer to or above ANN baseline
4. **Fair comparison**: ANN baseline also uses concept_dropout=0.3 and 50 epochs

### Open Items
- [ ] Retrain spiking CBM with new settings and compare against previous results
- [ ] Retrain ANN baseline with matching config
- [ ] Run intervention consistency on new checkpoints
- [ ] Run calibration on new checkpoints with proper 4-way split
- [ ] Unify duplicate hook code in `vmem_gate.py` (imports from `cbm.py` instead)
- [ ] Reconcile contradictory conclusions between `run_full_evaluation.py` and `run_full_evaluation_audit.py`
