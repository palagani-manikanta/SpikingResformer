# Prediction stability under tiny input noise (eps = 1e-06, 2 draws)

150 test images, CBM seed 0 (epoch 44), device cpu.

| Condition | Accuracy (%) | Spike-feature r vs clean | Class agreement vs clean (%) | Concept decisions changed (of 112) |
|:---|:---:|:---:|:---:|:---:|
| saved | 51.33 | 0.864 | 61.3 | 5.73 |
| clean | 54.00 | 1.000 | 100.0 | 0.00 |
| noise1 | 54.00 | 0.916 | 76.0 | 4.00 |
| noise2 | 57.33 | 0.915 | 71.3 | 4.30 |

Class changed under at least one noise draw: 38.0% (95% CI 30.6-46.0%).
Mean clean top-class probability: 0.641 (class stable) vs 0.364 (class changed).
Changed when clean confidence >= 0.5: 15.4% (n = 78); < 0.5: 62.5%.
