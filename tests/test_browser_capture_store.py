from __future__ import annotations

import pathlib
import types

import numpy as np

from local_ai.slices.voice.web_ui.capture_store import (
    append_capture_audio,
    close_capture_writer,
    ensure_capture_writer,
)
from local_ai.slices.voice.web_ui.session_state import SessionState


class FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self.params: list[tuple[str, object]] = []
        self.frames: list[bytes] = []

    def setnchannels(self, value: int) -> None:
        self.params.append(("channels", value))

    def setsampwidth(self, value: int) -> None:
        self.params.append(("sampwidth", value))

    def setframerate(self, value: int) -> None:
        self.params.append(("framerate", value))

    def writeframes(self, payload: bytes) -> None:
        self.frames.append(payload)

    def close(self) -> None:
        self.closed = True


def make_session(tmp_path: pathlib.Path, *, save_sample: bool = True) -> SessionState:
    return SessionState(
        session_id="abc",
        queue=__import__("asyncio").Queue(maxsize=1),
        save_sample=save_sample,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


def test_ensure_capture_writer_creates_writer(tmp_path: pathlib.Path, monkeypatch) -> None:
    session = make_session(tmp_path)
    writer = FakeWriter()
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.wave.open", lambda path, mode: writer)
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.time.time", lambda: 123)

    out_path = ensure_capture_writer(session)

    assert out_path == tmp_path / "captures" / "capture_abc_123.wav"
    assert session.capture_writer is writer
    assert session.capture_sample_rate == 16000
    assert ("channels", 1) in writer.params
    assert ("sampwidth", 2) in writer.params
    assert ("framerate", 16000) in writer.params


def test_append_capture_audio_writes_pcm16(tmp_path: pathlib.Path, monkeypatch) -> None:
    session = make_session(tmp_path)
    writer = FakeWriter()
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.pathlib.Path.cwd", lambda: tmp_path)
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.wave.open", lambda path, mode: writer)
    monkeypatch.setattr("local_ai.slices.voice.web_ui.capture_store.time.time", lambda: 123)

    out_path = append_capture_audio(session, np.array([0.0, 1.0], dtype=np.float32))

    assert out_path == tmp_path / "captures" / "capture_abc_123.wav"
    assert session.capture_samples == 2
    assert len(writer.frames) == 1
    assert writer.frames[0] == (np.array([0, 32767], dtype=np.int16)).tobytes()


def test_append_capture_audio_noops_when_disabled(tmp_path: pathlib.Path) -> None:
    session = make_session(tmp_path, save_sample=False)
    out_path = append_capture_audio(session, np.array([0.0], dtype=np.float32))
    assert out_path is None
    assert session.capture_samples == 0


def test_close_capture_writer_returns_path_and_clears_writer(tmp_path: pathlib.Path) -> None:
    session = make_session(tmp_path)
    writer = FakeWriter()
    session.capture_writer = writer
    session.capture_path = tmp_path / "capture.wav"

    out_path = close_capture_writer(session)

    assert out_path == tmp_path / "capture.wav"
    assert session.capture_writer is None
    assert writer.closed is True
