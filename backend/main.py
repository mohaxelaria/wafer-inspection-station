"""Web tier: operator console, control API, live telemetry.

Runs in two shapes, chosen by whether REDIS_URL is set:

* **standalone** (no REDIS_URL) - this process also owns the machine. One
  command, no services: `uvicorn backend.main:app --reload`.
* **distributed** (REDIS_URL set) - the machine lives in `backend.tool_service`;
  this process only subscribes to Redis, serves the UI and forwards commands.
  Holding no machine state is what allows several replicas of this container.

    docker compose up --build
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .bus import make_bus
from .db import make_store
from .runtime import MachineRunner, persist

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

bus = make_bus()
store = make_store()
DISTRIBUTED = os.environ.get("REDIS_URL") is not None

# In standalone mode this process owns the machine; in distributed mode the
# tool container does, and `runner` stays None.
runner: MachineRunner | None = None
clients: set[WebSocket] = set()


async def _fan_out() -> None:
    """Forward bus events to every connected browser."""
    async for evt in bus.events():
        if DISTRIBUTED:
            persist(store, evt)      # standalone persists inside the runner
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
    global runner
    coros = [_fan_out()]
    if not DISTRIBUTED:
        runner = MachineRunner(bus, store, persist_results=True)
        coros += runner.tasks()
        detector = runner.machine.detector.name
    else:
        detector = "in the tool service"
    print(f"[web] mode={'distributed' if DISTRIBUTED else 'standalone'}  "
          f"bus={bus.name}  store={store.backend}  detector={detector}")

    tasks = [asyncio.create_task(c) for c in coros]
    yield
    for t in tasks:
        t.cancel()
    await bus.close()


app = FastAPI(title="Wafer Inspection Station", version="0.3.0",
              lifespan=lifespan)


# ---------------------------------------------------------------- control API

@app.post("/api/command/{name}")
async def command(name: str, wafers: int = 5, grid: int = 26):
    if name not in {"load", "start", "pause", "abort", "clear"}:
        raise HTTPException(404, f"unknown command: {name}")
    params = {"wafers": wafers, "grid": grid}

    if runner is not None:                     # standalone: act directly
        try:
            return runner.dispatch(name, params)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))

    await bus.send_command({"name": name, "params": params})
    return {"accepted": True, "name": name}    # the tool applies it


@app.get("/api/state")
async def state():
    if runner is not None:
        return runner.machine.snapshot()
    snapshot = await bus.get_state()
    if snapshot is None:
        raise HTTPException(503, "no machine state yet - is the tool running?")
    return snapshot


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
    detector = runner.machine.detector.name if runner else "tool-service"
    return {"status": "ok",
            "mode": "distributed" if DISTRIBUTED else "standalone",
            "bus": bus.name, "store": store.backend, "detector": detector}


# ------------------------------------------------------------------ websocket

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        snapshot = (runner.machine.snapshot() if runner
                    else await bus.get_state())
        if snapshot:
            await ws.send_text(json.dumps({"type": "snapshot", **snapshot}))
        while True:
            await ws.receive_text()            # client keepalive only
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="ui")
