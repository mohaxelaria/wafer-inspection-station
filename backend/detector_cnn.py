"""CNN detector: the WM-811K model plugged into the running machine.

Imports torch lazily so the rest of the application still starts on a machine
with no ML stack installed. If the checkpoint is missing, `load_detector()`
falls back to the heuristic, and the UI shows which one is live.
"""
from __future__ import annotations

from pathlib import Path

from .detector import Detector, HeuristicDetector
from .wafer import Wafer, wafer_to_grid

CKPT_DIR = Path(__file__).resolve().parent.parent / "ml" / "checkpoints"


def best_checkpoint(directory: Path = CKPT_DIR) -> Path | None:
    """Pick the checkpoint with the highest validation macro F1.

    Each input resolution trains its own file (wafer_cnn_32.pt,
    wafer_cnn_64.pt, ...), so after an experiment the machine simply runs
    whichever one scored best - no config to remember to change.
    """
    import torch

    best, best_f1 = None, -1.0
    for path in sorted(directory.glob("wafer_cnn*.pt")):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            f1 = float(blob.get("val_macro_f1", 0.0))
        except Exception:
            continue
        if f1 > best_f1:
            best, best_f1 = path, f1
    return best


class CNNDetector:
    """Wraps the trained classifier behind the same interface as the heuristic."""

    def __init__(self, checkpoint: Path | None = None):
        import numpy as np
        import torch

        from ml.model import WaferCNN

        checkpoint = checkpoint or best_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoint in {CKPT_DIR}")
        self._np = np
        self._torch = torch
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes: list[str] = list(blob["classes"])
        self.size: int = int(blob["size"])
        self.model = WaferCNN(n_classes=len(self.classes))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        f1 = blob.get("val_macro_f1")
        self.name = (f"cnn-wm811k-{self.size}px(f1={f1:.2f})" if f1
                     else f"cnn-wm811k-{self.size}px")

    def _tensor(self, wafer: Wafer):
        np, torch = self._np, self._torch
        grid = np.asarray(wafer_to_grid(wafer), dtype=np.uint8)
        h, w = grid.shape
        rows = (np.arange(self.size) * h // self.size).clip(0, h - 1)
        cols = (np.arange(self.size) * w // self.size).clip(0, w - 1)
        small = grid[rows][:, cols]
        x = np.stack([(small == 1), (small == 2)]).astype("float32")
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def classify(self, wafer: Wafer) -> tuple[str, float]:
        torch = self._torch
        with torch.no_grad():
            probs = torch.softmax(self.model(self._tensor(wafer)), dim=1)[0]
        idx = int(probs.argmax())
        return self.classes[idx], float(probs[idx])


def load_detector() -> Detector:
    """Prefer the trained model; fall back to the heuristic with a clear note."""
    try:
        if best_checkpoint() is not None:
            det = CNNDetector()
            print(f"[detector] loaded {det.name} on {det.device}")
            return det
        print("[detector] no checkpoint yet; using the heuristic")
    except Exception as exc:
        print(f"[detector] CNN unavailable ({exc}); using the heuristic")
    return HeuristicDetector()
