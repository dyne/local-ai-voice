from __future__ import annotations

import asyncio


async def event_stream(*, queue: asyncio.Queue[str], ping_timeout: float = 15.0) -> object:
    while True:
        try:
            line = await asyncio.wait_for(queue.get(), timeout=ping_timeout)
            yield f"data: {line}\n\n"
        except asyncio.TimeoutError:
            yield "event: ping\ndata: keepalive\n\n"
        except asyncio.CancelledError:
            break
