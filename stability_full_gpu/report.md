# Prediction stability under tiny input noise (eps = 1e-06, 2 draws)

5794 test images, CBM seed 0 (epoch 44), device cuda.

| Condition | Accuracy (%) | Spike-feature r vs clean | Class agreement vs clean (%) | Concept decisions changed (of 112) |
|:---|:---:|:---:|:---:|:---:|
| saved | 58.77 | 1.000 | 100.0 | 0.00 |
| clean | 58.77 | 1.000 | 100.0 | 0.00 |
| noise1 | 59.84 | 0.917 | 78.3 | 3.95 |
| noise2 | 58.70 | 0.916 | 78.4 | 3.97 |

Class changed under at least one noise draw: 31.3% (95% CI 30.1-32.5%).
Mean clean top-class probability: 0.658 (class stable) vs 0.366 (class changed).
Changed when clean confidence >= 0.5: 10.2% (n = 3157); < 0.5: 56.5%.
