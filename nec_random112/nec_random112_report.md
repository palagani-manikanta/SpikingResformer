# NEC with a size-matched random baseline (112 random neurons)

Protocol identical to `nec_sparse/nec_sparse_report.md` (VLG-CBM sparse layer, strict NEC, FISTA, alpha 0.99); only the random CBL size changes from 512 to 112, the size of our concept layer. Mean +- std over seeds 0-2; gap = trained concepts minus random-112 at NEC = 5, pooled paired bootstrap over test images (10,000 resamples).

| Model | Trained 112 concepts ANEC-5 | Random-112 ANEC-5 | Random-512 ANEC-5 | Gap vs random-112 (95% CI) |
|---|---|---|---|---|
| SNN + GRU (ours) | 42.52 ± 0.74 | 43.91 ± 0.56 | 44.25 ± 1.59 | -1.39 [-2.19, -0.60] |
| ResNet-34 + MLP | 44.82 ± 0.95 | 47.78 ± 0.63 | 48.56 ± 1.43 | -2.96 [-3.73, -2.21] |
| ResNet-18 + MLP | 41.43 ± 1.23 | 44.78 ± 1.73 | 46.12 ± 0.06 | -3.35 [-4.07, -2.62] |

Reading: a CI entirely above 0 means the trained concepts beat a random layer of the same size at NEC = 5; a CI containing 0 means no detectable difference; entirely below 0 means random wins.
