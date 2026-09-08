"""Turn a raw wafer-map grid into the `Wafer` object the rest of the code uses.

This is what lets the heuristic detector - written against the simulator - be
scored on real WM-811K maps without changing a line of it.
"""
from __future__ import annotations

from backend.wafer import Die, Wafer


def wafer_from_grid(grid, lot_id: str = "WM811K", slot: int = 1,
                    label: str = "none") -> Wafer:
    """grid[r][c]: 0 = no die, 1 = pass, 2 = fail (the WM-811K encoding)."""
    rows = len(grid)
    cols = len(grid[0])
    hr, hc = (rows - 1) / 2 or 1, (cols - 1) / 2 or 1
    dies = []
    for r in range(rows):
        for c in range(cols):
            v = int(grid[r][c])
            if v == 0:
                continue
            dies.append(Die(row=r, col=c, x=(c - hc) / hc, y=(r - hr) / hr,
                            failed=(v == 2), inspected=True))
    return Wafer(wafer_id=f"{lot_id}-{slot}", lot_id=lot_id, slot=slot,
                 grid=max(rows, cols), dies=dies, true_pattern=label)
