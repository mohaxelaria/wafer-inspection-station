"""SQLite result store.

Milestone 1 keeps this deliberately small and dependency-free. The schema is
the one PostgreSQL will get in milestone 2 - only the driver changes.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

# Override with WIS_DB_PATH when the project lives on a network/shared mount,
# where SQLite file locking is not available.
DB_PATH = Path(os.environ.get(
    "WIS_DB_PATH",
    Path(__file__).resolve().parent.parent / "data" / "inspection.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    lot_id      TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    finished_at REAL,
    wafer_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wafers (
    wafer_id     TEXT PRIMARY KEY,
    lot_id       TEXT NOT NULL,
    slot         INTEGER NOT NULL,
    dies_total   INTEGER NOT NULL,
    dies_failed  INTEGER NOT NULL,
    yield_pct    REAL NOT NULL,
    true_pattern TEXT NOT NULL,
    pred_pattern TEXT NOT NULL,
    confidence   REAL NOT NULL,
    detector     TEXT NOT NULL,
    duration_s   REAL NOT NULL,
    finished_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alarms (
    alarm_id  TEXT PRIMARY KEY,
    code      TEXT NOT NULL,
    severity  TEXT NOT NULL,
    text      TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wafers_lot ON wafers(lot_id);
"""


class Store:
    backend = "sqlite"

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def add_lot(self, lot_id: str, wafer_count: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO lots (lot_id, started_at, wafer_count)"
                " VALUES (?,?,?)", (lot_id, time.time(), wafer_count))
            self._conn.commit()

    def finish_lot(self, lot_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE lots SET finished_at=? WHERE lot_id=?",
                               (time.time(), lot_id))
            self._conn.commit()

    def add_wafer(self, evt: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO wafers (wafer_id, lot_id, slot,"
                " dies_total, dies_failed, yield_pct, true_pattern,"
                " pred_pattern, confidence, detector, duration_s, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (evt["wafer_id"], evt["lot_id"], evt["slot"], evt["dies_total"],
                 evt["dies_failed"], evt["yield_pct"], evt["true_pattern"],
                 evt["pred_pattern"], evt["confidence"], evt["detector"],
                 evt["duration_s"], time.time()))
            self._conn.commit()

    def add_alarm(self, evt: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO alarms (alarm_id, code, severity, text, ts)"
                " VALUES (?,?,?,?,?)",
                (evt["alarm_id"], evt["code"], evt["severity"], evt["text"],
                 evt["ts"]))
            self._conn.commit()

    def recent_wafers(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM wafers ORDER BY finished_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def detector_accuracy(self) -> dict:
        """How often the detector's label matched the simulator's ground truth."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT true_pattern, pred_pattern FROM wafers").fetchall()
        total = len(rows)
        hits = sum(1 for r in rows if r["true_pattern"] == r["pred_pattern"])
        confusion: dict[str, dict[str, int]] = {}
        for r in rows:
            confusion.setdefault(r["true_pattern"], {})
            confusion[r["true_pattern"]][r["pred_pattern"]] = (
                confusion[r["true_pattern"]].get(r["pred_pattern"], 0) + 1)
        return {"wafers": total,
                "accuracy": round(hits / total, 3) if total else 0.0,
                "confusion": confusion}

    def lot_summary(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT l.lot_id, l.started_at, l.finished_at, l.wafer_count,"
                " COUNT(w.wafer_id) AS done, AVG(w.yield_pct) AS avg_yield"
                " FROM lots l LEFT JOIN wafers w ON w.lot_id = l.lot_id"
                " GROUP BY l.lot_id ORDER BY l.started_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]


def make_store():
    """PostgreSQL when DATABASE_URL is set, SQLite otherwise.

    Both classes expose the same methods, so the machine and the API never
    learn which one they are talking to.
    """
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        from .db_postgres import PostgresStore
        return PostgresStore(dsn)
    return Store()
