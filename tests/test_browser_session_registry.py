from __future__ import annotations

import asyncio

import pytest

from local_ai.slices.voice.web_ui.session_registry import (
    close_unknown_session,
    replace_existing_session,
    reset_existing_audio_socket_session,
)
from local_ai.slices.voice.web_ui.session_state import SessionState


class FakeSocket:
    def __init__(self) -> None:
        self.closed: list[tuple[int | None, str | None]] = []

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append((code, reason))


def make_session(session_id: str = "abc") -> SessionState:
    return SessionState(
        session_id=session_id,
        queue=asyncio.Queue(maxsize=1),
        save_sample=False,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


@pytest.mark.anyio
async def test_close_unknown_session_returns_true_and_closes_socket() -> None:
    websocket = FakeSocket()
    handled = await close_unknown_session(None, websocket)

    assert handled is True
    assert websocket.closed == [(4404, "Unknown session")]


@pytest.mark.anyio
async def test_close_unknown_session_returns_false_when_session_exists() -> None:
    handled = await close_unknown_session(make_session(), FakeSocket())
    assert handled is False


@pytest.mark.anyio
async def test_replace_existing_session_calls_cleanup_when_present() -> None:
    sessions = {"abc": make_session("abc")}
    cleaned: list[str] = []

    async def cleanup(session: SessionState) -> None:
        cleaned.append(session.session_id)
        sessions.pop(session.session_id, None)

    await replace_existing_session("abc", sessions, cleanup)

    assert cleaned == ["abc"]
    assert sessions == {}


@pytest.mark.anyio
async def test_reset_existing_audio_socket_session_returns_none_after_cleanup() -> None:
    session = make_session("abc")
    session.audio_socket = FakeSocket()
    sessions = {"abc": session}
    websocket = FakeSocket()

    async def cleanup(current: SessionState) -> None:
        sessions.pop(current.session_id, None)

    result = await reset_existing_audio_socket_session("abc", session, sessions, cleanup, websocket)

    assert result is None
    assert websocket.closed == [(4409, "Session reset")]
