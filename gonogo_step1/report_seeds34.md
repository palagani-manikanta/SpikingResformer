# Go/no-go step 1: concept loss (seeds 3, 4)

2 seeds, epochs selected on held-out class accuracy. NEC = 5 with the VLG-CBM sparse layer; random-112 = 112 random directions on the same variant's features. Gaps: paired bootstrap over test images (10,000 resamples, pooled over seeds), 95% CI.

PASS = (a) SNN trained concepts beat random-112 at NEC = 5 (CI > 0) **and** (b) SNN accuracy within 1 point of ResNet-34 + MLP with the same variant (CI of SNN - ANN > -1).

| Variant | SNN acc | ANN acc | SNN - ANN (95% CI) | Concept errors % SNN / ANN | SNN NEC-5 trained vs random-112 | SNN gap vs random (95% CI) | ANN NEC-5 trained vs random-112 | Verdict |
|---|---|---|---|---|---|---|---|---|
| base | 58.69 | 58.23 | +0.46 [-0.72, +1.66] | 12.59 / 10.85 | 42.63 vs 42.79 | -0.16 [-1.09, +0.77] | 45.18 vs 47.83 | fail (-b) |
| lc3 | 57.58 | 57.67 | -0.09 [-1.31, +1.06] | 8.82 / 7.80 | 46.00 vs 44.11 | +1.90 [+0.96, +2.81] | 51.26 vs 48.44 | fail (a-) |

base = the published joint runs (lambda_concept 1). Variants: lc5 / lc20 = concept loss weight 5 / 20; lc5pw = 5 + rare-concept weighting; seq = sequential CBM (concepts first, then a head on frozen concepts).
