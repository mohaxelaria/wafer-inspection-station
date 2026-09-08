"""Wafer geometry and synthetic defect-pattern generation.

The pattern taxonomy follows the WM-811K wafer map dataset, so the class
labels stay valid when a model trained on WM-811K replaces the heuristic
detector in backend/detector.py.
"""
from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field

# WM-811K uses: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch,
# Near-full, none. We start with the six that are visually distinct on a
# coarse grid; the rest can be added once the CNN is in place.
PATTERNS = ["none", "center", "donut", "edge_ring", "scratch", "loc"]

# How often each pattern shows up in a simulated lot. Real fabs are mostly
# clean, so "none" dominates.
PATTERN_WEIGHTS = [0.45, 0.11, 0.09, 0.13, 0.12, 0.10]


@dataclass
class Die:
    row: int
    col: int
    x: float          # normalised wafer coordinates, -1.0 .. 1.0
    y: float
    failed: bool = False
    inspected: bool = False

    @property
    def r(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass
class Wafer:
    wafer_id: str
    lot_id: str
    slot: int                      # 1..25, the cassette slot
    grid: int
    dies: list[Die]
    true_pattern: str              # ground truth, for scoring the detector
    pred_pattern: str | None = None
    pred_confidence: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def total(self) -> int:
        return len(self.dies)

    @property
    def inspected(self) -> int:
        return sum(1 for d in self.dies if d.inspected)

    @property
    def failed(self) -> int:
        return sum(1 for d in self.dies if d.inspected and d.failed)

    @property
    def yield_pct(self) -> float:
        done = self.inspected
        if done == 0:
            return 0.0
        return 100.0 * (done - self.failed) / done


def _pattern_probability(pattern: str, die: Die, rng: random.Random,
                         line: tuple[float, float, float] | None,
                         blob: tuple[float, float] | None) -> float:
    """Probability that this die fails, given the wafer's defect pattern."""
    r = die.r
    if pattern == "center":
        return 0.92 * math.exp(-((r / 0.34) ** 2))
    if pattern == "donut":
        return 0.90 * math.exp(-(((r - 0.58) / 0.13) ** 2))
    if pattern == "edge_ring":
        return 0.88 if r > 0.80 else 0.02 * max(0.0, r - 0.4)
    if pattern == "scratch" and line is not None:
        a, b, c = line                      # a*x + b*y + c = 0, normalised
        dist = abs(a * die.x + b * die.y + c)
        return 0.90 * math.exp(-((dist / 0.055) ** 2))
    if pattern == "loc" and blob is not None:
        bx, by = blob
        d = math.hypot(die.x - bx, die.y - by)
        return 0.88 * math.exp(-((d / 0.20) ** 2))
    return 0.0


def build_wafer(lot_id: str, slot: int, grid: int = 26,
                pattern: str | None = None,
                base_fail_rate: float = 0.012,
                seed: int | None = None) -> Wafer:
    """Create one wafer with a hidden defect pattern the detector must find."""
    rng = random.Random(seed)
    if pattern is None:
        pattern = rng.choices(PATTERNS, weights=PATTERN_WEIGHTS, k=1)[0]

    # A scratch is a random chord across the wafer; a Loc defect is a blob.
    line = None
    blob = None
    if pattern == "scratch":
        theta = rng.uniform(0, math.pi)
        a, b = math.cos(theta), math.sin(theta)
        c = rng.uniform(-0.55, 0.55)
        line = (a, b, c)
    elif pattern == "loc":
        rho = rng.uniform(0.30, 0.70)
        phi = rng.uniform(0, 2 * math.pi)
        blob = (rho * math.cos(phi), rho * math.sin(phi))

    dies: list[Die] = []
    half = (grid - 1) / 2.0
    for row in range(grid):
        for col in range(grid):
            x = (col - half) / half
            y = (row - half) / half
            if math.hypot(x, y) > 1.0:       # outside the wafer edge
                continue
            die = Die(row=row, col=col, x=x, y=y)
            p = _pattern_probability(pattern, die, rng, line, blob)
            p = max(p, base_fail_rate)       # background random failures
            die.failed = rng.random() < p
            dies.append(die)

    # Scanning order: serpentine, row by row, like a real stage.
    dies.sort(key=lambda d: (d.row, d.col if d.row % 2 == 0 else -d.col))

    return Wafer(
        wafer_id=uuid.uuid4().hex[:12],
        lot_id=lot_id,
        slot=slot,
        grid=grid,
        dies=dies,
        true_pattern=pattern,
    )
