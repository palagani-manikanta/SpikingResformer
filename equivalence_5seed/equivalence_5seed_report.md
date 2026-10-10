# GRU vs ResNet-34 + MLP: accuracy equivalence with 5 seeds

Generated 2026-10-10 14:00:13 by `equivalence_5seed.py`. Same protocol as `accuracy_equivalence.py` (paired bootstrap over the 5,794 test images, 10,000 resamples; TOST at alpha 0.05 with the pre-set margin ±1 pt), with seeds 0, 1, 2, 3, 4 instead of 0–2.

| Selection | Seeds | Gap (pts) | 95% CI | Equivalent within ±1 pt? (bootstrap / seed-t) | Smallest margin shown (bootstrap / seed-t) |
|:---|:---:|:---:|:---:|:---:|:---:|
| acc | 0–2 (published) | -0.10 | [-1.20, +1.01] | no / no | 1.02 / 1.32 pt |
| acc | **0–4** | +0.12 | [-0.92, +1.17] | no (p = 0.052) / yes (p = 0.015) | 1.01 / 0.69 pt |
| auc | 0–2 (published) | +0.09 | [-1.04, +1.22] | no / no | 1.04 / 1.37 pt |
| auc | **0–4** | -0.05 | [-1.10, +1.01] | yes (p = 0.038) / yes (p = 0.015) | 0.93 / 0.66 pt |

**Verdict:** Not equivalent within ±1 pt under every rule/interval; keep the wording "equivalent within about ±1.0 pt".

Per-seed gaps (class-acc selection): +0.38, +0.26, -0.93, +0.48, +0.43 pt.
