# Detector comparison on real WM-811K data

Held-out test split: 6,834 real wafer maps at 64x64, identical input for both detectors.

### Heuristic (hand-written features)

accuracy **0.402** &middot; macro F1 **0.214**

| class | support | recall | F1 |
|---|---:|---:|---:|
| none | 3000 | 0.36 | 0.42 |
| center | 645 | 0.04 | 0.08 |
| donut | 84 | 0.33 | 0.44 |
| edge_loc | 779 | 0.00 | 0.00 |
| edge_ring | 1452 | 0.99 | 0.56 |
| loc | 540 | 0.17 | 0.23 |
| random | 131 | 0.00 | 0.00 |
| scratch | 180 | 0.49 | 0.20 |
| near_full | 23 | 0.00 | 0.00 |

### CNN trained on WM-811K (64x64 input)

accuracy **0.933** &middot; macro F1 **0.877**

| class | support | recall | F1 |
|---|---:|---:|---:|
| none | 3000 | 0.97 | 0.96 |
| center | 645 | 0.94 | 0.92 |
| donut | 84 | 0.87 | 0.85 |
| edge_loc | 779 | 0.85 | 0.86 |
| edge_ring | 1452 | 0.96 | 0.97 |
| loc | 540 | 0.80 | 0.83 |
| random | 131 | 0.96 | 0.88 |
| scratch | 180 | 0.86 | 0.83 |
| near_full | 23 | 0.87 | 0.78 |
