"""Offline benchmark for whichever detector is wired into the machine.

    python -m scripts.bench_detector [n_per_pattern]

Generates wafers with known patterns, marks every die inspected, and prints
overall accuracy plus a confusion matrix with per-class recall. This is the
number the WM-811K CNN has to beat.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

from backend.detector import HeuristicDetector
from backend.wafer import PATTERNS, build_wafer


def main(n: int = 120) -> None:
    det = HeuristicDetector()
    confusion: dict[str, Counter] = defaultdict(Counter)
    hits = total = 0

    for seed, pattern in enumerate(PATTERNS * n):
        wafer = build_wafer("BENCH", slot=1, pattern=pattern, seed=seed)
        for die in wafer.dies:
            die.inspected = True
        pred, _ = det.classify(wafer)
        confusion[pattern][pred] += 1
        total += 1
        hits += pred == pattern

    width = max(len(p) for p in PATTERNS) + 2
    print(f"\ndetector: {det.name}   wafers: {total}   "
          f"accuracy: {hits / total:.1%}\n")
    print(" " * width + "".join(f"{p:>11}" for p in PATTERNS))
    for truth in PATTERNS:
        row = confusion[truth]
        cells = "".join(f"{row.get(p, 0):>11}" for p in PATTERNS)
        recall = row.get(truth, 0) / max(sum(row.values()), 1)
        print(f"{truth:<{width}}{cells}   recall {recall:>4.0%}")
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
