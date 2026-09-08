"""Simulated optical inspection tool.

Stands in for a real SP-series inspection machine: it holds a cassette of
wafers, drives a stage over each die, reports telemetry at 10 Hz, raises
alarms, and can be driven with the same start/pause/abort commands a real
host would send. Everything downstream (backend, HMI) is written against
this interface, so a real tool could replace it without touching the UI.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Awaitable, Callable

from .detector import Detector, HeuristicDetector
from .wafer import Wafer, build_wafer

TICK = 0.1                     # seconds; 10 Hz telemetry, like a real tool


class State(str, Enum):
    IDLE = "IDLE"
    LOADING = "LOADING"
    SCANNING = "SCANNING"
    PAUSED = "PAUSED"
    FAULT = "FAULT"


@dataclass
class Alarm:
    code: str
    severity: str              # warning | serious | critical
    text: str
    ts: float = field(default_factory=time.time)
    alarm_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    active: bool = True


@dataclass
class Telemetry:
    state: str = State.IDLE.value
    lot_id: str | None = None
    slot: int = 0
    slots_total: int = 0
    stage_x: float = 0.0
    stage_y: float = 0.0
    lamp_pct: float = 100.0
    chamber_c: float = 22.0
    vacuum_kpa: float = 0.0
    dies_done: int = 0
    dies_total: int = 0
    dies_failed: int = 0
    throughput_dph: float = 0.0     # dies per hour, rolling
    uptime_s: float = 0.0


Emit = Callable[[dict], None]


class InspectionMachine:
    def __init__(self, detector: Detector | None = None, emit: Emit | None = None,
                 dies_per_second: float = 55.0, seed: int | None = None):
        self.detector = detector or HeuristicDetector()
        self.emit = emit or (lambda evt: None)
        self.dies_per_second = dies_per_second
        self.rng = random.Random(seed)

        self.state = State.IDLE
        self.telemetry = Telemetry()
        self.alarms: list[Alarm] = []

        self.lot_id: str | None = None
        self.wafers: list[Wafer] = []
        self.wafer_index: int = -1
        self._cursor: int = 0          # index into current wafer's die list
        self._carry: float = 0.0       # fractional dies left from last tick
        self._t0 = time.time()
        self._recent: list[tuple[float, int]] = []   # (ts, dies) for throughput
        self._misalign_until: float = 0.0
        self._task: asyncio.Task | None = None

    # ---------------------------------------------------------------- commands

    def load_lot(self, wafer_count: int = 5, grid: int = 26) -> str:
        if self.state in (State.SCANNING, State.LOADING):
            raise RuntimeError("machine is busy")
        self.lot_id = "LOT" + uuid.uuid4().hex[:6].upper()
        self.wafers = [build_wafer(self.lot_id, slot=i + 1, grid=grid)
                       for i in range(wafer_count)]
        self.wafer_index = -1
        self._set_state(State.LOADING)
        self.emit({"type": "lot_loaded", "lot_id": self.lot_id,
                   "wafer_count": wafer_count})
        return self.lot_id

    def start(self) -> None:
        if self.state == State.FAULT:
            raise RuntimeError("clear the fault first")
        if not self.wafers:
            self.load_lot()
        if self.state in (State.LOADING, State.IDLE):
            self._next_wafer()
        elif self.state == State.PAUSED:
            self._set_state(State.SCANNING)

    def pause(self) -> None:
        if self.state == State.SCANNING:
            self._set_state(State.PAUSED)

    def abort(self) -> None:
        self.wafers = []
        self.lot_id = None
        self.wafer_index = -1
        self._set_state(State.IDLE)

    def clear_alarms(self) -> None:
        for a in self.alarms:
            a.active = False
        if self.state == State.FAULT:
            self._set_state(State.PAUSED)
        self.emit({"type": "alarms_cleared"})

    # ------------------------------------------------------------------ engine

    async def run(self) -> None:
        """Main loop. Started once by the FastAPI lifespan handler."""
        while True:
            try:
                self._tick()
            except Exception as exc:                      # never kill the loop
                self._raise_alarm("SIM_ERROR", "critical", f"simulator: {exc}")
            await asyncio.sleep(TICK)

    def _tick(self) -> None:
        now = time.time()
        self.telemetry.uptime_s = now - self._t0

        # Lamp ages while scanning and is the tool's slow-degradation signal.
        if self.state == State.SCANNING:
            self.telemetry.lamp_pct = max(60.0, self.telemetry.lamp_pct - 0.004)
            self.telemetry.chamber_c += self.rng.uniform(-0.03, 0.035)
            self.telemetry.vacuum_kpa = 92.0 + self.rng.uniform(-0.6, 0.6)
        else:
            self.telemetry.chamber_c += (22.0 - self.telemetry.chamber_c) * 0.02
            self.telemetry.vacuum_kpa *= 0.9

        if self.telemetry.lamp_pct < 85.0 and not self._has_active("LAMP_DEGRADED"):
            self._raise_alarm("LAMP_DEGRADED", "warning",
                              "Illumination below 85% - schedule lamp change")

        if self.state == State.LOADING and now > self._misalign_until:
            self._next_wafer()

        if self.state == State.SCANNING:
            self._scan_step(now)

        self.telemetry.state = self.state.value
        self.emit({"type": "telemetry", **asdict(self.telemetry)})

    def _scan_step(self, now: float) -> None:
        wafer = self.wafers[self.wafer_index]

        # Rare hard fault: the stage fails to reach position.
        if self.rng.random() < 0.0006:
            self._raise_alarm("STAGE_TIMEOUT", "critical",
                              f"Stage move timeout at die {self._cursor}")
            self._set_state(State.FAULT)
            return

        self._carry += self.dies_per_second * TICK
        count = int(self._carry)
        self._carry -= count
        if count <= 0:
            return

        batch = []
        for _ in range(count):
            if self._cursor >= len(wafer.dies):
                break
            die = wafer.dies[self._cursor]
            die.inspected = True
            self._cursor += 1
            self.telemetry.stage_x = die.x
            self.telemetry.stage_y = die.y
            batch.append({"row": die.row, "col": die.col,
                          "failed": bool(die.failed)})

        self.telemetry.dies_done = wafer.inspected
        self.telemetry.dies_failed = wafer.failed
        self._recent.append((now, len(batch)))
        self._recent = [(t, n) for t, n in self._recent if now - t < 10.0]
        window = sum(n for _, n in self._recent)
        self.telemetry.throughput_dph = window * 360.0     # per 10 s -> per hour

        if batch:
            self.emit({"type": "dies", "slot": wafer.slot, "dies": batch})

        if self._cursor >= len(wafer.dies):
            self._finish_wafer(wafer)

    # ------------------------------------------------------------------ wafers

    def _next_wafer(self) -> None:
        self.wafer_index += 1
        if self.wafer_index >= len(self.wafers):
            self.emit({"type": "lot_done", "lot_id": self.lot_id})
            self.wafers = []
            self.wafer_index = -1
            self._set_state(State.IDLE)
            return

        wafer = self.wafers[self.wafer_index]
        wafer.started_at = time.time()
        self._cursor = 0
        self._carry = 0.0
        self.telemetry.lot_id = self.lot_id
        self.telemetry.slot = wafer.slot
        self.telemetry.slots_total = len(self.wafers)
        self.telemetry.dies_total = wafer.total
        self.telemetry.dies_done = 0
        self.telemetry.dies_failed = 0

        # Occasionally the wafer is not seated correctly and needs re-aligning.
        if self.rng.random() < 0.06:
            self._raise_alarm("WAFER_MISALIGN", "serious",
                              f"Slot {wafer.slot} misaligned - re-centering")
            self._misalign_until = time.time() + 2.0
            self._set_state(State.LOADING)
            return

        self._set_state(State.SCANNING)
        self.emit({"type": "wafer_started", "slot": wafer.slot,
                   "grid": wafer.grid, "dies_total": wafer.total,
                   "lot_id": self.lot_id})

    def _finish_wafer(self, wafer: Wafer) -> None:
        wafer.finished_at = time.time()
        label, conf = self.detector.classify(wafer)
        wafer.pred_pattern, wafer.pred_confidence = label, conf
        self.emit({
            "type": "wafer_done",
            "wafer_id": wafer.wafer_id,
            "lot_id": wafer.lot_id,
            "slot": wafer.slot,
            "dies_total": wafer.total,
            "dies_failed": wafer.failed,
            "yield_pct": round(wafer.yield_pct, 2),
            "true_pattern": wafer.true_pattern,
            "pred_pattern": label,
            "confidence": round(conf, 3),
            "detector": self.detector.name,
            "duration_s": round(wafer.finished_at - (wafer.started_at or 0), 1),
        })
        self._set_state(State.LOADING)
        self._misalign_until = time.time() + 0.8      # cassette exchange time

    # ------------------------------------------------------------------ helpers

    def _set_state(self, state: State) -> None:
        if state != self.state:
            self.state = state
            self.emit({"type": "state", "state": state.value})

    def _has_active(self, code: str) -> bool:
        return any(a.code == code and a.active for a in self.alarms)

    def _raise_alarm(self, code: str, severity: str, text: str) -> None:
        alarm = Alarm(code=code, severity=severity, text=text)
        self.alarms.append(alarm)
        self.alarms = self.alarms[-50:]
        self.emit({"type": "alarm", **asdict(alarm)})

    def snapshot(self) -> dict:
        wafer = (self.wafers[self.wafer_index]
                 if 0 <= self.wafer_index < len(self.wafers) else None)
        return {
            "telemetry": asdict(self.telemetry),
            "alarms": [asdict(a) for a in self.alarms[-20:]],
            "wafer": None if wafer is None else {
                "slot": wafer.slot,
                "grid": wafer.grid,
                "dies_total": wafer.total,
                "dies": [{"row": d.row, "col": d.col, "failed": bool(d.failed)}
                         for d in wafer.dies if d.inspected],
            },
        }
