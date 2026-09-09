"""Wiring between the machine, the bus and the result store.

Used by both deployment shapes:

* `backend.tool_service` - the equipment-side container, which owns the machine
  and publishes to Redis.
* `backend.main` in standalone mode - one process doing everything, for local
  development without Docker.

Keeping the wiring here means the two entry points cannot drift apart.
"""
from __future__ import annotations

import asyncio

from .bus import InProcessBus
from .detector_cnn import load_detector
from .machine import InspectionMachine

STATE_INTERVAL = 1.0        # seconds between machine snapshots on the bus


def persist(store, evt: dict) -> None:
    """Durable side effects for the events that carry results."""
    try:
        kind = evt.get("type")
        if kind == "lot_loaded":
            store.add_lot(evt["lot_id"], evt["wafer_count"])
        elif kind == "wafer_done":
            store.add_wafer(evt)
        elif kind == "lot_done":
            store.finish_lot(evt["lot_id"])
        elif kind == "alarm":
            store.add_alarm(evt)
    except Exception as exc:
        print(f"[store] {exc}")


class MachineRunner:
    """Owns one InspectionMachine and connects it to a bus and a store."""

    def __init__(self, bus, store, persist_results: bool = True):
        self.bus = bus
        self.store = store
        self.persist_results = persist_results
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=4000)
        self.machine = InspectionMachine(detector=load_detector(),
                                         emit=self._emit)

    # The machine calls this synchronously from its tick, so it must not block.
    def _emit(self, evt: dict) -> None:
        try:
            self._queue.put_nowait(evt)
        except asyncio.QueueFull:
            pass                      # telemetry is disposable, results are not

    async def _pump(self) -> None:
        while True:
            evt = await self._queue.get()
            if self.persist_results:
                persist(self.store, evt)
            await self.bus.publish(evt)

    async def _state_loop(self) -> None:
        while True:
            await self.bus.set_state(self.machine.snapshot())
            await asyncio.sleep(STATE_INTERVAL)

    async def _command_loop(self) -> None:
        async for cmd in self.bus.commands():
            try:
                self.dispatch(cmd.get("name", ""), cmd.get("params") or {})
            except Exception as exc:
                print(f"[command] {cmd}: {exc}")

    def dispatch(self, name: str, params: dict | None = None) -> dict:
        """Execute one operator command. Raises ValueError on an unknown name."""
        params = params or {}
        if name == "load":
            return {"lot_id": self.machine.load_lot(
                wafer_count=int(params.get("wafers", 5)),
                grid=int(params.get("grid", 26)))}
        if name == "start":
            self.machine.start()
        elif name == "pause":
            self.machine.pause()
        elif name == "abort":
            self.machine.abort()
        elif name == "clear":
            self.machine.clear_alarms()
        else:
            raise ValueError(f"unknown command: {name}")
        return {"ok": True, "state": self.machine.state.value}

    def tasks(self) -> list:
        """Coroutines the host process must run for the machine to live."""
        return [self.machine.run(), self._pump(), self._state_loop(),
                self._command_loop()]
