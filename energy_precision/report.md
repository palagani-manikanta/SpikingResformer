# Energy with one precision for compute and memory

SNN (stem once): 2.675 G AC + 124.1 M MAC per image. Memory bytes: reference scenario of energy_memory (16-bit membrane) at the stated width -- SNN 282.2 MB (8-bit) / 588.2 MB (32-bit). ANN MACs from energy_audit_v2, bytes from energy_memory. Ratio = ANN energy / SNN energy (> 1: SNN cheaper). Break-even: memory pJ/B above which the SNN costs more.

| Precision | ANN | SNN compute (mJ) | ANN compute (mJ) | Ratio, compute only | Ratio, 1 MB SRAM (12.5 pJ/B) | Ratio, DRAM low (162.5 pJ/B) | Ratio, DRAM high (325 pJ/B) | Break-even (pJ/B) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Published: FP32 compute + 8-bit memory (mixed) | ResNet-34 + MLP 512-1841-512 (capacity-matched) | 2.978 | 16.860 | 5.66x | 2.65x | 0.45x | 0.28x | 55.1 |
| Published: FP32 compute + 8-bit memory (mixed) | ResNet-18 + MLP 512-1841-512 (capacity-matched) | 2.978 | 8.351 | 2.80x | 1.32x | 0.23x | 0.15x | 20.3 |
| Published: FP32 compute + 8-bit memory (mixed) | ResNet-50 + linear CBM | 2.978 | 18.802 | 6.31x | 2.98x | 0.54x | 0.35x | 66.9 |
| FP32 compute + 32-bit memory | ResNet-34 + MLP 512-1841-512 (capacity-matched) | 2.978 | 16.860 | 5.66x | 1.78x | 0.37x | 0.29x | 29.8 |
| FP32 compute + 32-bit memory | ResNet-18 + MLP 512-1841-512 (capacity-matched) | 2.978 | 8.351 | 2.80x | 0.89x | 0.20x | 0.16x | 10.4 |
| FP32 compute + 32-bit memory | ResNet-50 + linear CBM | 2.978 | 18.802 | 6.31x | 2.04x | 0.49x | 0.40x | 39.0 |
| INT8, 32-bit accumulate + 8-bit memory | ResNet-34 + MLP 512-1841-512 (capacity-matched) | 0.305 | 1.100 | 3.61x | 0.39x | 0.13x | 0.12x | 3.2 |
| INT8, 32-bit accumulate + 8-bit memory | ResNet-18 + MLP 512-1841-512 (capacity-matched) | 0.305 | 0.545 | 1.79x | 0.20x | 0.07x | 0.07x | 0.9 |
| INT8, 32-bit accumulate + 8-bit memory | ResNet-50 + linear CBM | 0.305 | 1.226 | 4.02x | 0.47x | 0.19x | 0.17x | 3.9 |
| INT8, 8-bit accumulate + 8-bit memory | ResNet-34 + MLP 512-1841-512 (capacity-matched) | 0.109 | 0.843 | 7.75x | 0.34x | 0.13x | 0.12x | 2.9 |
| INT8, 8-bit accumulate + 8-bit memory | ResNet-18 + MLP 512-1841-512 (capacity-matched) | 0.109 | 0.418 | 3.84x | 0.18x | 0.07x | 0.07x | 1.2 |
| INT8, 8-bit accumulate + 8-bit memory | ResNet-50 + linear CBM | 0.109 | 0.940 | 8.64x | 0.42x | 0.18x | 0.17x | 3.5 |

Notes:
- The first row reproduces the published numbers (5.66x compute, 2.65x 1 MB SRAM, 0.45x/0.28x DRAM, break-even 55.1 pJ/B) and mixes 32-bit arithmetic with 8-bit memory.
- Rows 2-4 are consistent: the arithmetic and the memory use the same precision.
- Op energies: Horowitz ISSCC 2014, 45 nm. First-order model, no cache hierarchy; not a hardware measurement.
