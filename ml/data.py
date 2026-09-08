"""WM-811K loading and preprocessing.

Download the dataset first (about 1 GB):

    pip install kaggle
    kaggle datasets download -d qingyi/wm811k-wafer-map -p data --unzip

That produces data/LSWMD.pkl. Then build the cached tensors:

    python -m ml.data --size 32 --max-per-class 20000

Only ~173k of the 811k wafers carry a failure label, and ~85% of those are
"none", so the cache subsamples the majority class and the training script
also weights the loss. Both steps are needed - either one alone still gives a
model that mostly predicts "none".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import CLASSES, FROM_WM811K, INDEX

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "LSWMD.pkl"


def _unwrap(value):
    """LSWMD stores labels as nested numpy arrays; some rows are empty."""
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    return value if isinstance(value, str) else None


def _resize_nearest(grid: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize. Wafer maps are categorical (0/1/2), so any
    interpolating resize would invent die states that never existed."""
    h, w = grid.shape
    rows = (np.arange(size) * h // size).clip(0, h - 1)
    cols = (np.arange(size) * w // size).clip(0, w - 1)
    return grid[rows][:, cols]


def build_cache(size: int = 32, max_per_class: int = 20000,
                seed: int = 0) -> Path:
    if not RAW.exists():
        raise SystemExit(
            f"{RAW} not found.\n"
            "Download it first:\n"
            "  kaggle datasets download -d qingyi/wm811k-wafer-map "
            "-p data --unzip")

    print(f"reading {RAW.name} ...")
    df = pd.read_pickle(RAW)
    df["label"] = df["failureType"].apply(_unwrap).map(FROM_WM811K)
    df = df[df["label"].notna()].reset_index(drop=True)
    print(f"labelled wafers: {len(df):,}")

    rng = np.random.default_rng(seed)
    keep = []
    for name in CLASSES:
        idx = np.flatnonzero(df["label"].values == name)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
        print(f"  {name:<10} {len(idx):>7,}")
    keep = np.sort(np.concatenate(keep))

    maps, labels = [], []
    for i in keep:
        grid = np.asarray(df["waferMap"].values[i], dtype=np.uint8)
        if grid.ndim != 2 or min(grid.shape) < 4:
            continue
        small = _resize_nearest(grid, size)
        maps.append(np.stack([(small == 1), (small == 2)]).astype(np.uint8))
        labels.append(INDEX[df["label"].values[i]])

    X = np.stack(maps)                      # (N, 2, size, size) pass / fail
    y = np.asarray(labels, dtype=np.int64)
    print(f"tensor: {X.shape}  ({X.nbytes / 1e6:.0f} MB)")

    # Stratified 70 / 15 / 15 split.
    idx_train, idx_val, idx_test = [], [], []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_tr = int(0.70 * len(idx))
        n_va = int(0.15 * len(idx))
        idx_train.append(idx[:n_tr])
        idx_val.append(idx[n_tr:n_tr + n_va])
        idx_test.append(idx[n_tr + n_va:])
    splits = {k: np.sort(np.concatenate(v)) for k, v in
              (("train", idx_train), ("val", idx_val), ("test", idx_test))}
    for k, v in splits.items():
        print(f"  {k:<6} {len(v):>7,}")

    out = ROOT / "data" / f"wm811k_{size}.npz"
    np.savez_compressed(out, X=X, y=y, classes=np.array(CLASSES),
                        **{f"idx_{k}": v for k, v in splits.items()})
    print(f"wrote {out}")
    return out


def load_cache(size: int = 32):
    path = ROOT / "data" / f"wm811k_{size}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python -m ml.data --size {size}")
    z = np.load(path, allow_pickle=False)
    return (z["X"], z["y"],
            {k: z[f"idx_{k}"] for k in ("train", "val", "test")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=20000)
    args = ap.parse_args()
    build_cache(args.size, args.max_per_class)
