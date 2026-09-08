"""Defect-pattern classifiers.

`Detector` is the interface the machine talks to. Milestone 1 ships
`HeuristicDetector`: hand-written geometric features over the failing dies, so
the whole pipeline runs today without a trained model. Milestone 2 adds
`CNNDetector` trained on WM-811K behind the same interface - nothing else in
the system has to change.

Run `python -m scripts.bench_detector` to score whichever detector is wired in.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Protocol

from .wafer import Wafer


class Detector(Protocol):
    name: str

    def classify(self, wafer: Wafer) -> tuple[str, float]:
        """Return (pattern label, confidence 0..1) for an inspected wafer."""
        ...


def _clusters(fails: list) -> list[list]:
    """8-connected clusters of failing dies, largest first.

    Real defects are contiguous; the background failure rate is not. Clustering
    first is what keeps random speckle from destroying the shape features.
    """
    index = {(d.row, d.col): d for d in fails}
    seen: set[tuple[int, int]] = set()
    out: list[list] = []
    for key in index:
        if key in seen:
            continue
        group, queue = [], deque([key])
        seen.add(key)
        while queue:
            r, c = queue.popleft()
            group.append(index[(r, c)])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in index and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
        out.append(group)
    out.sort(key=len, reverse=True)
    return out


def _shape(group: list) -> tuple[float, float]:
    """(elongation, spread) from the second moment of a die cluster."""
    n = len(group)
    mx = sum(d.x for d in group) / n
    my = sum(d.y for d in group) / n
    sxx = sum((d.x - mx) ** 2 for d in group) / n
    syy = sum((d.y - my) ** 2 for d in group) / n
    sxy = sum((d.x - mx) * (d.y - my) for d in group) / n
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = max(tr * tr / 4 - det, 0.0)
    l1 = tr / 2 + math.sqrt(disc)
    l2 = tr / 2 - math.sqrt(disc)
    elongation = math.sqrt(l1 / l2) if l2 > 1e-9 else 99.0
    return elongation, math.sqrt(max(l1, 0.0))


def features(wafer: Wafer) -> dict[str, float]:
    """Geometric summary of the failing dies on an inspected wafer."""
    done = [d for d in wafer.dies if d.inspected]
    fails = [d for d in done if d.failed]
    n = len(fails)
    if n == 0 or not done:
        return {"n": 0.0, "rate": 0.0, "big_size": 0.0}

    radii = [d.r for d in fails]
    mean_r = sum(radii) / n
    var_r = sum((r - mean_r) ** 2 for r in radii) / n

    groups = _clusters(fails)
    big = groups[0]
    elongation, spread = _shape(big) if len(big) >= 3 else (1.0, 0.0)

    return {
        "n": float(n),
        "rate": n / len(done),
        "mean_r": mean_r,
        "std_r": math.sqrt(var_r),
        "edge_frac": sum(1 for r in radii if r > 0.78) / n,
        "core_frac": sum(1 for r in radii if r < 0.35) / n,
        "big_size": float(len(big)),
        "big_share": len(big) / n,
        "big_mean_r": sum(d.r for d in big) / len(big),
        "elongation": elongation,
        "spread": spread,
        "n_clusters": float(len(groups)),
    }


class HeuristicDetector:
    """Rule-based baseline. Strong enough to prove the pipeline end to end,
    weak enough to make the case for the CNN that replaces it."""

    name = "heuristic-v1"

    def classify(self, wafer: Wafer) -> tuple[str, float]:
        f = features(wafer)

        # Nothing but background speckle: no cluster worth naming.
        if f["n"] < 10 or f["rate"] < 0.025 or f["big_size"] < 5:
            return "none", 0.82

        # Radial position separates the ring families from everything else.
        if f["edge_frac"] > 0.55:
            return "edge_ring", min(0.55 + f["edge_frac"] / 2, 0.95)
        if f["core_frac"] > 0.45:
            return "center", min(0.55 + f["core_frac"] / 2, 0.95)

        # A scratch is the only pattern that is strongly elongated.
        if f["elongation"] > 2.6 and f["big_size"] >= 8:
            return "scratch", min(0.55 + f["elongation"] / 20, 0.95)

        # Donut: a wide annulus - large, spread out, hollow in the middle.
        if (f["spread"] > 0.30 and f["big_size"] > 40
                and f["core_frac"] < 0.15 and f["edge_frac"] < 0.35):
            return "donut", min(0.60 + f["spread"] / 4, 0.92)

        # Loc: one compact blob holding most of the failures.
        if f["spread"] < 0.30 and f["big_share"] > 0.35:
            return "loc", min(0.55 + f["big_share"] / 3, 0.92)

        return "none", 0.45
