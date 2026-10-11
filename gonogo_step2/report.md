# Go/no-go step 2: attention readout (5 seeds, lambda_c = 3)

| Model | Mean pooling (step 1) | Attention | Gain (95% CI) | NEC-5 concepts - random-112 (attention) |
|---|---|---|---|---|
| SNN + GRU | 57.65 | 61.16 | +3.51 [+2.68, +4.35] | +5.46 [+4.79, +6.14] |
| ResNet-34 + MLP | 57.29 | 60.29 | +3.00 [+2.32, +3.68] | +2.83 [+2.24, +3.41] |

SNN - ANN with attention: +0.87 [-0.10, +1.85]

Rule (fixed before running): (1) SNN attention gain CI > 0; (2) SNN concepts beat random-112 (CI > 0); (3) SNN - ANN (attention) CI > -1.

Verdict: (1) pass, (2) pass, (3) pass -> **PASS**
