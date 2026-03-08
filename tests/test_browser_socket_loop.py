from __future__ import annotations

import asyncio

import numpy as np
import pytest

from local_ai.slices.voice.web_ui.message_processor import MessageProcessingResult
from local_ai.slices.voice.web_ui.session_state import SessionState
from local_ai.slices.voice.web_ui.socket_loop import handle_audio_socket_connection


class FakeDisconnect(Exception):
    pass


class FakeWebSocket:
    def __init__(self, messages: list[object]) -> None:
        self.messages = list(messages)
        self.accepted = False
        self.closed: list[tuple[int | None, str | None]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> object:
        item = self.messages.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append((code, reason))


def make_session(session_id: str = "abc", *, debug: bool = False) -> SessionState:
    return SessionState(
        session_id=session_id,
        queue=asyncio.Queue(maxsize=10),
        save_sample=False,
        silence_detect=True,
        debug=debug,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


@pytest.mark.anyio
async def test_handle_audio_socket_connection_accepts_and_processes_messages() -> None:
    session = make_session(debug=True)
    sessions = {"abc": session}
    websocket = FakeWebSocket([{"bytes": b"abc"}])
    debug_messages: list[str] = []
    cleanup_calls: list[str] = []
    processed: list[tuple[object, list[float]]] = []

    async def debug_fn(current: SessionState, message: str) -> None:
        debug_messages.append(message)

    async def cleanup_session_fn(current: SessionState) -> None:
        cleanup_calls.append(current.session_id)

    async def process_message_fn(*, session: SessionState, message: object, buffered_audio: np.ndarray) -> MessageProcessingResult:
        processed.append((message, buffered_audio.tolist()))
        return MessageProcessingResult(buffered_audio=np.asarray([0.25], dtype=np.float32), stop=True)

    await handle_audio_socket_connection(
        session_id="abc",
        websocket=websocket,
        sessions=sessions,
        cleanup_session_fn=cleanup_session_fn,
        debug_fn=debug_fn,
        process_message_fn=process_message_fn,
        websocket_disconnect_type=FakeDisconnect,
    )

    assert websocket.accepted is True
    assert session.audio_socket is websocket
    assert session.received_messages == 1
    assert processed == [({"bytes": b"abc"}, [])]
    assert cleanup_calls == ["abc"]
    assert debug_messages == ["audio websocket connected", "received blob #1 bytes=3"]


@pytest.mark.anyio
async def test_handle_audio_socket_connection_ignores_disconnect_and_cleans_up() -> None:
    session = make_session()
    websocket = FakeWebSocket([FakeDisconnect()])
    cleanup_calls: list[str] = []

    async def cleanup_session_fn(current: SessionState) -> None:
        cleanup_calls.append(current.session_id)

    async def process_message_fn(*, session: SessionState, message: object, buffered_audio: np.ndarray) -> MessageProcessingResult:
        raise AssertionError("should not process after disconnect")

    await handle_audio_socket_connection(
        session_id="abc",
        websocket=websocket,
        sessions={"abc": session},
        cleanup_session_fn=cleanup_session_fn,
        debug_fn=lambda session, message: asyncio.sleep(0),
        process_message_fn=process_message_fn,
        websocket_disconnect_type=FakeDisconnect,
    )

    assert cleanup_calls == ["abc"]


@pytest.mark.anyio
async def test_handle_audio_socket_connection_reports_stream_error() -> None:
    session = make_session()
    websocket = FakeWebSocket([{"bytes": b"abc"}, RuntimeError("boom")])
    cleanup_calls: list[str] = []

    async def cleanup_session_fn(current: SessionState) -> None:
        cleanup_calls.append(current.session_id)

    async def process_message_fn(*, session: SessionState, message: object, buffered_audio: np.ndarray) -> MessageProcessingResult:
        return MessageProcessingResult(buffered_audio=np.asarray([], dtype=np.float32), stop=False)

    await handle_audio_socket_connection(
        session_id="abc",
        websocket=websocket,
        sessions={"abc": session},
        cleanup_session_fn=cleanup_session_fn,
        debug_fn=lambda session, message: asyncio.sleep(0),
        process_message_fn=process_message_fn,
        websocket_disconnect_type=FakeDisconnect,
    )

    assert cleanup_calls == ["abc"]
    assert await session.queue.get() == "[server error] Audio stream failed: boom"
