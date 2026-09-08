"""FastAPI host process: machine control, result storage, live telemetry.

Run with:  uvicorn backend.main:app --reload
Then open: http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .db import Store
from .detector_cnn import load_detector
from .machine import InspectionMachine

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

store = Store()
events: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)
clients: set[WebSocket] = set()


def _emit(evt: dict) -> None:
    """Called by the machine on every tick. Must never block the loop."""
    try:
        events.put_nowait(evt)
    except asyncio.QueueFull:
        pass                       # telemetry is disposable; results are not


machine = InspectionMachine(detector=load_detector(), emit=_emit)


async def _broadcaster() -> None:
    while True:
        evt = await events.get()

        # Durable side effects before the event reaches any screen.
        try:
            if evt["type"] == "lot_loaded":
                store.add_lot(evt["lot_id"], evt["wafer_count"])
            elif evt["type"] == "wafer_done":
                store.add_wafer(evt)
            elif evt["type"] == "lot_done":
                store.finish_lot(evt["lot_id"])
            elif evt["type"] == "alarm":
                store.add_alarm(evt)
        except Exception as exc:
            print(f"[store] {exc}")

        if not clients:
            continue
        payload = json.dumps(evt)
        for ws in list(clients):
            try:
                await ws.send_text(payload)
            except Exception:
                clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(machine.run()),
             asyncio.create_task(_broadcaster())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Wafer Inspection Station", version="0.1.0",
              lifespan=lifespan)


# ---------------------------------------------------------------- control API

@app.post("/api/command/{name}")
async def command(name: str, wafers: int = 5, grid: int = 26):
    try:
        if name == "load":
            return {"lot_id": machine.load_lot(wafer_count=wafers, grid=grid)}
        if name == "start":
            machine.start()
        elif name == "pause":
            machine.pause()
        elif name == "abort":
            machine.abort()
        elif name == "clear":
            machine.clear_alarms()
        else:
            raise HTTPException(404, f"unknown command: {name}")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"ok": True, "state": machine.state.value}


@app.get("/api/state")
async def state():
    return machine.snapshot()


# ---------------------------------------------------------------- results API

@app.get("/api/wafers")
async def wafers(limit: int = 50):
    return store.recent_wafers(limit)


@app.get("/api/lots")
async def lots():
    return store.lot_summary()


@app.get("/api/metrics/detector")
async def detector_metrics():
    return store.detector_accuracy()


@app.get("/api/health")
async def health():
    return {"status": "ok", "state": machine.state.value,
            "detector": machine.detector.name}


# ------------------------------------------------------------------ websocket

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "snapshot", **machine.snapshot()}))
        while True:
            await ws.receive_text()          # client keepalive only
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
