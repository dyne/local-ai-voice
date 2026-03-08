from __future__ import annotations

import asyncio

import pytest

from local_ai.slices.voice.web_ui.event_stream import event_stream


@pytest.mark.anyio
async def test_event_stream_yields_queue_messages() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue()
    await queue.put("hello")

    stream = event_stream(queue=queue, ping_timeout=1.0)
    first = await anext(stream)

    assert first == "data: hello\n\n"


@pytest.mark.anyio
async def test_event_stream_yields_keepalive_on_timeout() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue()

    stream = event_stream(queue=queue, ping_timeout=0.01)
    first = await anext(stream)

    assert first == "event: ping\ndata: keepalive\n\n"


@pytest.mark.anyio
async def test_event_stream_stops_on_cancellation() -> None:
    queue: asyncio.Queue[str] = asyncio.Queue()
    stream = event_stream(queue=queue, ping_timeout=1.0)
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(StopAsyncIteration):
        await task
