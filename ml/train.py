"""Train the wafer-map classifier on WM-811K.

    python -m ml.train --epochs 20

Writes ml/checkpoints/wafer_cnn.pt, which backend/detector_cnn.py picks up
automatically the next time the server starts.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_cache
from .labels import CLASSES
from .model import WaferCNN

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def ckpt_path(size: int) -> Path:
    """One checkpoint per input resolution, so a 64x64 run never overwrites
    the 32x32 result it is being compared against."""
    return CKPT_DIR / f"wafer_cnn_{size}.pt"


def augment(x: torch.Tensor) -> torch.Tensor:
    """Wafer defect classes are invariant to flips and 90-degree rotations."""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(x, k, dims=[-2, -1])


def macro_f1(conf: np.ndarray) -> tuple[float, list[float]]:
    per_class = []
    for c in range(len(CLASSES)):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        per_class.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(per_class)), per_class


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray]:
    model.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu()
        for t, p in zip(yb.numpy(), pred.numpy()):
            conf[t, p] += 1
        correct += int((pred == yb).sum())
        total += len(yb)
    return correct / max(total, 1), conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()
    ckpt = ckpt_path(args.size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    X, y, splits = load_cache(args.size)
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y)

    def loader(name, shuffle):
        idx = splits[name]
        return DataLoader(TensorDataset(Xt[idx], yt[idx]),
                          batch_size=args.batch, shuffle=shuffle,
                          num_workers=0, drop_last=False)

    train_dl, val_dl, test_dl = (loader("train", True), loader("val", False),
                                 loader("test", False))

    counts = np.bincount(y[splits["train"]], minlength=len(CLASSES))
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5,
                           dtype=torch.float32, device=device)
    weights /= weights.mean()
    print("class counts:", dict(zip(CLASSES, counts.tolist())))

    model = WaferCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for xb, yb in train_dl:
            xb = augment(xb).to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach().item() * len(yb)

        acc, conf = evaluate(model, val_dl, device)
        f1, _ = macro_f1(conf)
        flag = ""
        if f1 > best:
            best = f1
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "classes": CLASSES, "size": args.size,
                        "val_macro_f1": f1, "val_acc": acc}, ckpt)
            flag = "  <- saved"
        print(f"epoch {epoch:>2}/{args.epochs}  "
              f"loss {running / len(splits['train']):.4f}  "
              f"val acc {acc:.3f}  val macroF1 {f1:.3f}  "
              f"{time.time() - t0:.0f}s{flag}")

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    acc, conf = evaluate(model, test_dl, device)
    f1, per_class = macro_f1(conf)
    print(f"\nTEST  accuracy {acc:.3f}   macro F1 {f1:.3f}\n")
    for name, score, n in zip(CLASSES, per_class, conf.sum(1)):
        print(f"  {name:<10} F1 {score:.3f}   n={n}")
    print(f"\ncheckpoint: {ckpt}")


if __name__ == "__main__":
    main()
