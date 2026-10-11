# Go/no-go step 1: concept loss

3 seeds, epochs selected on held-out class accuracy. NEC = 5 with the VLG-CBM sparse layer; random-112 = 112 random directions on the same variant's features. Gaps: paired bootstrap over test images (10,000 resamples, pooled over seeds), 95% CI.

PASS = (a) SNN trained concepts beat random-112 at NEC = 5 (CI > 0) **and** (b) SNN accuracy within 1 point of ResNet-34 + MLP with the same variant (CI of SNN - ANN > -1).

| Variant | SNN acc | ANN acc | SNN - ANN (95% CI) | Concept errors % SNN / ANN | SNN NEC-5 trained vs random-112 | SNN gap vs random (95% CI) | ANN NEC-5 trained vs random-112 | Verdict |
|---|---|---|---|---|---|---|---|---|
| base | 58.60 | 58.69 | -0.10 [-1.20, +1.01] | 12.54 / 9.78 | 42.52 vs 43.91 | -1.39 [-2.18, -0.60] | 44.82 vs 47.78 | fail (--) |
| lc2 | 57.31 | 58.05 | -0.74 [-1.85, +0.38] | 10.28 / 8.67 | 44.22 vs 43.17 | +1.05 [+0.25, +1.86] | 48.16 vs 48.03 | fail (a-) |
| lc3 | 57.70 | 57.04 | +0.66 [-0.45, +1.79] | 8.81 / 7.99 | 45.89 vs 41.92 | +3.97 [+3.15, +4.83] | 51.29 vs 47.28 | **PASS** (ab) |
| lc5 | 55.62 | 56.90 | -1.28 [-2.39, -0.17] | 8.46 / 7.53 | 46.59 vs 42.39 | +4.20 [+3.39, +5.05] | 52.64 vs 46.86 | fail (a-) |
| lc20 | 54.73 | 56.47 | -1.74 [-2.87, -0.60] | 7.99 / 7.21 | 47.09 vs 42.41 | +4.68 [+3.88, +5.51] | 53.60 vs 47.60 | fail (a-) |
| lc5pw | 55.65 | 57.56 | -1.91 [-3.05, -0.80] | 9.26 / 7.87 | 47.32 vs 41.57 | +5.76 [+4.95, +6.58] | 53.32 vs 46.51 | fail (a-) |
| seq | 53.80 | 55.68 | -1.88 [-3.03, -0.72] | 7.75 / 7.09 | 46.92 vs 41.70 | +5.22 [+4.41, +6.04] | 52.96 vs 47.20 | fail (a-) |

base = the published joint runs (lambda_concept 1). Variants: lc5 / lc20 = concept loss weight 5 / 20; lc5pw = 5 + rare-concept weighting; seq = sequential CBM (concepts first, then a head on frozen concepts).
