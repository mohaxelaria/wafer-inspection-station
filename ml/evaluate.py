"""Head-to-head: heuristic vs CNN on the same real WM-811K test split.

    python -m ml.evaluate            # 32x32 input
    python -m ml.evaluate --size 64  # after training at 64x64

Writes ml/RESULTS.md. This comparison is the point of milestone 2: the
heuristic looks excellent on clean simulated wafers and much less excellent on
real ones, which is the honest argument for training a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backend.detector import HeuristicDetector
from .compat import wafer_from_grid
from .data import load_cache
from .labels import CLASSES

OUT_DIR = Path(__file__).resolve().parent


def grids_from_tensor(X: np.ndarray) -> np.ndarray:
    """(N,2,S,S) pass/fail channels -> (N,S,S) with the 0/1/2 encoding."""
    return (X[:, 0] * 1 + X[:, 1] * 2).astype(np.uint8)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    n = len(CLASSES)
    conf = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    f1s = []
    for c in range(n):
        tp = conf[c, c]
        denom = 2 * tp + (conf[:, c].sum() - tp) + (conf[c, :].sum() - tp)
        f1s.append(float(2 * tp / denom) if denom else 0.0)
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    return acc, float(np.mean(f1s)), conf


def run_heuristic(grids: np.ndarray) -> np.ndarray:
    det = HeuristicDetector()
    idx = {name: i for i, name in enumerate(CLASSES)}
    out = np.empty(len(grids), dtype=np.int64)
    for i, g in enumerate(grids):
        label, _ = det.classify(wafer_from_grid(g.tolist()))
        out[i] = idx.get(label, idx["none"])
    return out


def run_cnn(X: np.ndarray, size: int):
    try:
        import torch
        from .train import ckpt_path
        from .model import WaferCNN
    except Exception as exc:
        print(f"(skipping CNN: {exc})")
        return None
    ckpt = ckpt_path(size)
    if not ckpt.exists():
        print(f"(skipping CNN: {ckpt.name} not found - "
              f"run python -m ml.train --size {size} first)")
        return None

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = WaferCNN(n_classes=len(blob["classes"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.from_numpy(X[i:i + 512]).float().to(device)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def table(name: str, acc: float, f1: float, conf: np.ndarray) -> str:
    lines = [f"### {name}", "",
             f"accuracy **{acc:.3f}** &middot; macro F1 **{f1:.3f}**", "",
             "| class | support | recall | F1 |", "|---|---:|---:|---:|"]
    for c, cls in enumerate(CLASSES):
        support = int(conf[c].sum())
        recall = conf[c, c] / support if support else 0.0
        tp = conf[c, c]
        denom = 2 * tp + (conf[:, c].sum() - tp) + (conf[c, :].sum() - tp)
        f1c = 2 * tp / denom if denom else 0.0
        lines.append(f"| {cls} | {support} | {recall:.2f} | {f1c:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32,
                    help="input resolution of the cache and checkpoint to score")
    args = ap.parse_args()

    X, y, splits = load_cache(args.size)
    test = splits["test"]
    Xt, yt = X[test], y[test]
    grids = grids_from_tensor(Xt)
    print(f"test wafers: {len(yt):,}   input {args.size}x{args.size}")

    sections = []
    h_acc, h_f1, h_conf = score(yt, run_heuristic(grids))
    print(f"heuristic  accuracy {h_acc:.3f}  macro F1 {h_f1:.3f}")
    sections.append(table("Heuristic (hand-written features)", h_acc, h_f1, h_conf))

    cnn_pred = run_cnn(Xt, args.size)
    if cnn_pred is not None:
        c_acc, c_f1, c_conf = score(yt, cnn_pred)
        print(f"CNN        accuracy {c_acc:.3f}  macro F1 {c_f1:.3f}")
        sections.append(table(f"CNN trained on WM-811K ({args.size}x{args.size} input)",
                              c_acc, c_f1, c_conf))

    out = OUT_DIR / ("RESULTS.md" if args.size == 32
                     else f"RESULTS_{args.size}.md")
    out.write_text(
        "# Detector comparison on real WM-811K data\n\n"
        f"Held-out test split: {len(yt):,} real wafer maps at "
        f"{args.size}x{args.size}, identical input for both detectors.\n\n"
        + "\n".join(sections), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
