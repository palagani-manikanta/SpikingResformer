# ResNet-34 + MLP with live augmentation

Same live views (TRAIN_TF parameters per seed, epoch and image), recipe, initialisation and selection (held-out class accuracy) as the spiking GRU CBM in `live_aug_order/`.

| Seed | SNN + GRU, live | ResNet-34 + MLP, live | ResNet-34 + MLP, 8 views | ANN live - 8 views |
|---|---|---|---|---|
| 0 | 60.08 | 56.01 | 58.39 | -2.38 |
| 1 | 59.87 | 58.73 | 58.11 | +0.62 |
| 2 | 60.23 | 59.51 | 59.58 | -0.07 |
| mean | 60.06 | 58.08 | 58.69 | -0.61 |

SNN + GRU minus ResNet-34 + MLP, both with live augmentation: +1.98 points, 95% CI [+0.86, +3.08] (paired bootstrap over test images, 10,000 resamples, pooled over seeds); TOST p (margin 1 point) = 0.956.
