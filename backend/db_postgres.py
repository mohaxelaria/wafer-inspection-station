"""PostgreSQL result store.

Same interface as the SQLite `Store` in db.py, so nothing above it changes.
Selected by setting DATABASE_URL, e.g.

    postgresql://wis:wis@postgres:5432/wis
"""
from __future__ import annotations

import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (
    lot_id      TEXT PRIMARY KEY,
    started_at  DOUBLE PRECISION NOT NULL,
    finished_at DOUBLE PRECISION,
    wafer_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS wafers (
    wafer_id     TEXT PRIMARY KEY,
    lot_id       TEXT NOT NULL,
    slot         INTEGER NOT NULL,
    dies_total   INTEGER NOT NULL,
    dies_failed  INTEGER NOT NULL,
    yield_pct    DOUBLE PRECISION NOT NULL,
    true_pattern TEXT NOT NULL,
    pred_pattern TEXT NOT NULL,
    confidence   DOUBLE PRECISION NOT NULL,
    detector     TEXT NOT NULL,
    duration_s   DOUBLE PRECISION NOT NULL,
    finished_at  DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS alarms (
    alarm_id  TEXT PRIMARY KEY,
    code      TEXT NOT NULL,
    severity  TEXT NOT NULL,
    text      TEXT NOT NULL,
    ts        DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wafers_lot ON wafers(lot_id);
CREATE INDEX IF NOT EXISTS idx_wafers_finished ON wafers(finished_at DESC);
"""


class PostgresStore:
    backend = "postgresql"

    def __init__(self, dsn: str, retries: int = 20):
        import psycopg
        from psycopg.rows import dict_row

        self._psycopg = psycopg
        last: Exception | None = None
        for attempt in range(retries):
            try:
                self._conn = psycopg.connect(dsn, autocommit=True,
                                             row_factory=dict_row)
                break
            except Exception as exc:          # container start ordering
                last = exc
                time.sleep(min(1 + attempt * 0.5, 5))
        else:
            raise RuntimeError(f"cannot reach PostgreSQL: {last}")

        with self._conn.cursor() as cur:
            cur.execute(SCHEMA)

    # ------------------------------------------------------------------ writes

    def add_lot(self, lot_id: str, wafer_count: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO lots (lot_id, started_at, wafer_count)"
                " VALUES (%s,%s,%s) ON CONFLICT (lot_id) DO NOTHING",
                (lot_id, time.time(), wafer_count))

    def finish_lot(self, lot_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute("UPDATE lots SET finished_at=%s WHERE lot_id=%s",
                        (time.time(), lot_id))

    def add_wafer(self, evt: dict) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO wafers (wafer_id, lot_id, slot, dies_total,"
                " dies_failed, yield_pct, true_pattern, pred_pattern,"
                " confidence, detector, duration_s, finished_at)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (wafer_id) DO NOTHING",
                (evt["wafer_id"], evt["lot_id"], evt["slot"], evt["dies_total"],
                 evt["dies_failed"], evt["yield_pct"], evt["true_pattern"],
                 evt["pred_pattern"], evt["confidence"], evt["detector"],
                 evt["duration_s"], time.time()))

    def add_alarm(self, evt: dict) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO alarms (alarm_id, code, severity, text, ts)"
                " VALUES (%s,%s,%s,%s,%s) ON CONFLICT (alarm_id) DO NOTHING",
                (evt["alarm_id"], evt["code"], evt["severity"], evt["text"],
                 evt["ts"]))

    # ------------------------------------------------------------------- reads

    def recent_wafers(self, limit: int = 50) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT * FROM wafers ORDER BY finished_at DESC"
                        " LIMIT %s", (limit,))
            return list(cur.fetchall())

    def detector_accuracy(self) -> dict:
        with self._conn.cursor() as cur:
            cur.execute("SELECT true_pattern, pred_pattern FROM wafers")
            rows = cur.fetchall()
        total = len(rows)
        hits = sum(1 for r in rows if r["true_pattern"] == r["pred_pattern"])
        confusion: dict[str, dict[str, int]] = {}
        for r in rows:
            bucket = confusion.setdefault(r["true_pattern"], {})
            bucket[r["pred_pattern"]] = bucket.get(r["pred_pattern"], 0) + 1
        return {"wafers": total,
                "accuracy": round(hits / total, 3) if total else 0.0,
                "confusion": confusion}

    def lot_summary(self, limit: int = 20) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT l.lot_id, l.started_at, l.finished_at, l.wafer_count,"
                " COUNT(w.wafer_id) AS done, AVG(w.yield_pct) AS avg_yield"
                " FROM lots l LEFT JOIN wafers w ON w.lot_id = l.lot_id"
                " GROUP BY l.lot_id ORDER BY l.started_at DESC LIMIT %s",
                (limit,))
            return list(cur.fetchall())
