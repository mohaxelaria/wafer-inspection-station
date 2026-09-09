"""Equipment-side process.

Owns the inspection tool, writes results to the database, and publishes every
event and a periodic state snapshot to Redis. It serves no HTTP: the web tier
subscribes to what this process publishes, which is why the web tier can be
scaled to several replicas while the machine stays single-instance - exactly
the constraint a real tool has.

    python -m backend.tool_service
"""
from __future__ import annotations

import asyncio
import os

from .bus import make_bus
from .db import make_store
from .runtime import MachineRunner


async def main() -> None:
    bus = make_bus()
    store = make_store()
    runner = MachineRunner(bus, store, persist_results=True)
    print(f"[tool] bus={bus.name}  store={store.backend}  "
          f"detector={runner.machine.detector.name}")

    if os.environ.get("AUTOSTART", "").lower() in ("1", "true", "yes"):
        runner.dispatch("load", {"wafers": int(os.environ.get("AUTOSTART_WAFERS", 25))})
        runner.dispatch("start")
        print("[tool] autostart: lot loaded and scanning")

    await asyncio.gather(*runner.tasks())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
