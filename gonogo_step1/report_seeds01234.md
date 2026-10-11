# Go/no-go step 1: concept loss (seeds 0, 1, 2, 3, 4)

5 seeds, epochs selected on held-out class accuracy. NEC = 5 with the VLG-CBM sparse layer; random-112 = 112 random directions on the same variant's features. Gaps: paired bootstrap over test images (10,000 resamples, pooled over seeds), 95% CI.

PASS = (a) SNN trained concepts beat random-112 at NEC = 5 (CI > 0) **and** (b) SNN accuracy within 1 point of ResNet-34 + MLP with the same variant (CI of SNN - ANN > -1).

| Variant | SNN acc | ANN acc | SNN - ANN (95% CI) | Concept errors % SNN / ANN | SNN NEC-5 trained vs random-112 | SNN gap vs random (95% CI) | ANN NEC-5 trained vs random-112 | Verdict |
|---|---|---|---|---|---|---|---|---|
| base | 58.63 | 58.51 | +0.12 [-0.92, +1.17] | 12.56 / 10.21 | 42.56 vs 43.46 | -0.89 [-1.55, -0.22] | 44.96 vs 47.80 | fail (-b) |
| lc3 | 57.65 | 57.29 | +0.36 [-0.71, +1.41] | 8.82 / 7.92 | 45.93 vs 42.79 | +3.14 [+2.44, +3.87] | 51.28 vs 47.74 | **PASS** (ab) |

base = the published joint runs (lambda_concept 1). Variants: lc5 / lc20 = concept loss weight 5 / 20; lc5pw = 5 + rare-concept weighting; seq = sequential CBM (concepts first, then a head on frozen concepts).
