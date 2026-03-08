from __future__ import annotations

import asyncio
import pathlib

import pytest

from local_ai.slices.voice.web_ui.session_cleanup import cleanup_session
from local_ai.slices.voice.web_ui.session_state import SessionState


class FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True


def make_session(session_id: str = "abc") -> SessionState:
    return SessionState(
        session_id=session_id,
        queue=asyncio.Queue(maxsize=10),
        save_sample=False,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


@pytest.mark.anyio
async def test_cleanup_session_closes_socket_and_removes_registry_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session()
    session.audio_socket = FakeSocket()
    sessions = {"abc": session}
    monkeypatch.setattr("local_ai.slices.voice.web_ui.session_cleanup.close_capture_writer", lambda current: None)

    await cleanup_session(session=session, sessions=sessions, target_sample_rate=16000)

    assert session.audio_socket is None
    assert sessions == {}


@pytest.mark.anyio
async def test_cleanup_session_emits_saved_capture_message(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session()
    session.capture_path = pathlib.Path("capture.wav")
    session.capture_sample_rate = 16000
    session.capture_samples = 8000
    sessions = {"abc": session}
    monkeypatch.setattr("local_ai.slices.voice.web_ui.session_cleanup.close_capture_writer", lambda current: current.capture_path)

    await cleanup_session(session=session, sessions=sessions, target_sample_rate=16000)

    assert session.queue.get_nowait() == "[server] saved WAV capture: capture.wav (0.50s)"


@pytest.mark.anyio
async def test_cleanup_session_tolerates_socket_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSocket:
        async def close(self, code: int | None = None, reason: str | None = None) -> None:
            raise RuntimeError("boom")

    session = make_session()
    session.audio_socket = BrokenSocket()
    sessions = {"abc": session}
    monkeypatch.setattr("local_ai.slices.voice.web_ui.session_cleanup.close_capture_writer", lambda current: None)

    await cleanup_session(session=session, sessions=sessions, target_sample_rate=16000)

    assert sessions == {}
