"""Shared label vocabulary.

WM-811K ships nine classes with mixed capitalisation and hyphens; the
simulator generates six of them. Everything inside this project uses the
snake_case names below so the two sources can be compared directly.
"""
from __future__ import annotations

# Canonical order - the model's output indices follow this list.
CLASSES = ["none", "center", "donut", "edge_loc", "edge_ring",
           "loc", "random", "scratch", "near_full"]

FROM_WM811K = {
    "none": "none",
    "Center": "center",
    "Donut": "donut",
    "Edge-Loc": "edge_loc",
    "Edge-Ring": "edge_ring",
    "Loc": "loc",
    "Random": "random",
    "Scratch": "scratch",
    "Near-full": "near_full",
}

INDEX = {name: i for i, name in enumerate(CLASSES)}
