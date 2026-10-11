# Analyses repeated on the improved model (lambda_c = 3 + attention readout, 5 seeds)

| Model | Accuracy | Concept AUC | ECE raw -> per-concept Platt | ICRC@25% raw -> calibrated head (gain, 95% CI) | ANEC-5 trained vs random-112 | ANEC-avg trained vs random-112 |
|---|---|---|---|---|---|---|
| SNN + GRU + attention | 61.16 ± 0.74 | 0.9548 ± 0.0008 | 0.0441 ± 0.0002 -> 0.0124 ± 0.0003 | 76.33 -> 80.87 (+4.54 [+4.34, +4.73]) | 53.51 vs 48.04 | 60.38 vs 58.38 |
| ResNet-34 + MLP + attention | 60.29 ± 0.69 | 0.9546 ± 0.0020 | 0.0407 ± 0.0006 -> 0.0143 ± 0.0003 | 73.85 -> 78.90 (+5.05 [+4.83, +5.26]) | 54.71 vs 51.88 | 58.52 vs 57.46 |

Calibrated-head ICRC@25%, SNN - ANN: +1.97 [+1.27, +2.68] points.

Intervention curve (mean over seeds, head retrained per arm), fractions 0.0, 0.1, 0.25, 0.5, 0.75, 1.0:
- SNN + GRU + attention, raw: 61.29, 67.23, 76.33, 93.59, 98.62, 98.98
- SNN + GRU + attention, per_concept_platt: 62.01, 70.00, 80.87, 96.09, 98.98, 98.67
- ResNet-34 + MLP + attention, raw: 60.40, 65.55, 73.85, 93.01, 98.66, 99.50
- ResNet-34 + MLP + attention, per_concept_platt: 60.43, 67.77, 78.90, 95.80, 99.11, 99.80

NEC curve (mean test accuracy at NEC = 5..30):
- SNN + GRU + attention, trained: 53.51, 61.52, 61.83, 61.75, 61.83, 61.83
- SNN + GRU + attention, random-112: 48.04, 58.96, 60.77, 60.79, 60.86, 60.86
- ResNet-34 + MLP + attention, trained: 54.71, 58.53, 59.24, 59.43, 59.61, 59.63
- ResNet-34 + MLP + attention, random-112: 51.88, 58.24, 58.44, 58.73, 58.73, 58.73

Energy: attention readout 0.25 uJ per image (210,352 AC + 12,544 MAC; tap spike density 0.019) vs 1.04 uJ for ResNet-34; totals 2.978 vs 16.861 mJ -> 5.66x (compute only, 45 nm, same rule as energy_audit_v2).
