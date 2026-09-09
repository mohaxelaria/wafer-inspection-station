# Detector comparison on real WM-811K data

Held-out test split: 6,834 real wafer maps, identical input for both detectors.

### Heuristic (hand-written features)

accuracy **0.476** &middot; macro F1 **0.235**

| class | support | recall | F1 |
|---|---:|---:|---:|
| none | 3000 | 0.55 | 0.58 |
| center | 645 | 0.04 | 0.07 |
| donut | 84 | 0.36 | 0.47 |
| edge_loc | 779 | 0.00 | 0.00 |
| edge_ring | 1452 | 0.96 | 0.57 |
| loc | 540 | 0.16 | 0.22 |
| random | 131 | 0.00 | 0.00 |
| scratch | 180 | 0.30 | 0.20 |
| near_full | 23 | 0.00 | 0.00 |

### CNN trained on WM-811K

accuracy **0.923** &middot; macro F1 **0.873**

| class | support | recall | F1 |
|---|---:|---:|---:|
| none | 3000 | 0.95 | 0.95 |
| center | 645 | 0.95 | 0.94 |
| donut | 84 | 0.89 | 0.89 |
| edge_loc | 779 | 0.85 | 0.86 |
| edge_ring | 1452 | 0.96 | 0.97 |
| loc | 540 | 0.80 | 0.80 |
| random | 131 | 0.95 | 0.88 |
| scratch | 180 | 0.73 | 0.70 |
| near_full | 23 | 0.91 | 0.86 |
