"""Message bus between the tool process and the web processes.

Two implementations behind one interface:

* `InProcessBus`  - everything in a single process (plain `uvicorn` for local
  development, no services needed).
* `RedisBus`      - the tool container publishes events and machine state to
  Redis; one or more web containers subscribe. This is what makes the web tier
  scalable: it holds no machine state of its own.

Channels and keys are deliberately few:
    wis.events    pub/sub  every machine event (telemetry, dies, alarms, ...)
    wis.commands  pub/sub  operator commands travelling the other way
    wis:state     key      last full snapshot, so a new client can render
                           immediately instead of waiting for the next tick
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

EVENTS = "wis.events"
COMMANDS = "wis.commands"
STATE_KEY = "wis:state"


class InProcessBus:
    """Single-process fallback: queues instead of a network hop."""

    name = "in-process"

    def __init__(self) -> None:
        self._event_subs: list[asyncio.Queue] = []
        self._command_subs: list[asyncio.Queue] = []
        self._state: dict | None = None

    async def publish(self, evt: dict) -> None:
        for q in list(self._event_subs):
            if q.full():
                continue                      # telemetry is disposable
            q.put_nowait(evt)

    async def events(self) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._event_subs.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._event_subs.remove(q)

    async def send_command(self, cmd: dict) -> None:
        for q in list(self._command_subs):
            q.put_nowait(cmd)

    async def commands(self) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._command_subs.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._command_subs.remove(q)

    async def set_state(self, state: dict) -> None:
        self._state = state

    async def get_state(self) -> dict | None:
        return self._state

    async def close(self) -> None:
        return None


class RedisBus:
    """Redis-backed bus. Only imported when REDIS_URL is set."""

    name = "redis"

    def __init__(self, url: str):
        import redis.asyncio as redis        # imported lazily on purpose

        self.url = url
        self._redis = redis.from_url(url, decode_responses=True)

    async def publish(self, evt: dict) -> None:
        await self._redis.publish(EVENTS, json.dumps(evt))

    async def events(self) -> AsyncIterator[dict]:
        async for msg in self._listen(EVENTS):
            yield msg

    async def send_command(self, cmd: dict) -> None:
        await self._redis.publish(COMMANDS, json.dumps(cmd))

    async def commands(self) -> AsyncIterator[dict]:
        async for msg in self._listen(COMMANDS):
            yield msg

    async def _listen(self, channel: str) -> AsyncIterator[dict]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    yield json.loads(message["data"])
                except (ValueError, TypeError):
                    continue                  # never let one bad frame stop us
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def set_state(self, state: dict) -> None:
        # Expires so a stale snapshot from a dead tool container cannot be
        # served as if the machine were still running.
        await self._redis.set(STATE_KEY, json.dumps(state), ex=30)

    async def get_state(self) -> dict | None:
        raw = await self._redis.get(STATE_KEY)
        return json.loads(raw) if raw else None

    async def close(self) -> None:
        await self._redis.aclose()


def make_bus():
    """RedisBus when REDIS_URL is set, InProcessBus otherwise."""
    url = os.environ.get("REDIS_URL")
    if url:
        return RedisBus(url)
    return InProcessBus()
