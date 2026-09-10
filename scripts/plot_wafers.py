"""Render real WM-811K wafer maps, one row per defect class.

    python -m scripts.plot_wafers --size 32 --per-class 6

Writes docs/wm811k_samples.png. Useful for the README, and for checking with
your own eyes that the classes look the way the labels claim.

A wafer map is not a photograph: every cell is one die, marked pass or fail by
electrical test. The CNN sees exactly what is drawn here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.data import load_cache
from ml.labels import CLASSES

OUT = Path(__file__).resolve().parent.parent / "docs" / "wm811k_samples.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
NO_DIE = "#f0efec"     # outside the wafer
PASS = "#c3c2b7"       # tested, good
FAIL = "#d03b3b"       # tested, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    X, y, splits = load_cache(args.size)
    test = splits["test"]
    X, y = X[test], y[test]
    rng = np.random.default_rng(args.seed)

    cmap = ListedColormap([NO_DIE, PASS, FAIL])
    rows, cols = len(CLASSES), args.per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.45),
                             facecolor=SURFACE)

    for r, name in enumerate(CLASSES):
        idx = np.flatnonzero(y == r)
        pick = rng.choice(idx, min(cols, len(idx)), replace=False)
        for c in range(cols):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c >= len(pick):
                ax.axis("off")
                continue
            # channel 0 = pass, channel 1 = fail -> 0 / 1 / 2
            grid = X[pick[c]][0] * 1 + X[pick[c]][1] * 2
            ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center",
                              fontsize=10, color=INK, labelpad=10)

    fig.suptitle(f"WM-811K wafer maps - real test data, {args.size}x{args.size} input",
                 fontsize=13, color=INK, y=0.995)
    fig.legend(handles=[Patch(facecolor=PASS, label="die passed"),
                        Patch(facecolor=FAIL, label="die failed"),
                        Patch(facecolor=NO_DIE, label="no die")],
               loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=(0, 0.022, 1, 0.985))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
