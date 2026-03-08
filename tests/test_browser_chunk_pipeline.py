from __future__ import annotations

import asyncio
import pathlib

import numpy as np
import pytest

from local_ai.slices.voice.web_ui.chunk_pipeline import process_prepared_chunks
from local_ai.slices.voice.web_ui.session_state import SessionState


def make_session() -> SessionState:
    return SessionState(
        session_id="abc",
        queue=asyncio.Queue(maxsize=20),
        save_sample=False,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


@pytest.mark.anyio
async def test_process_prepared_chunks_emits_transcript() -> None:
    session = make_session()
    debug_messages: list[str] = []

    async def fake_inference(**kwargs):
        return type("Result", (), {"text": "hello", "error": None})()

    await process_prepared_chunks(
        session=session,
        chunks=[np.array([0.1], dtype=np.float32)],
        target_sample_rate=16000,
        append_capture_audio_fn=lambda session, chunk: None,
        run_chunk_inference_fn=fake_inference,
        debug_fn=lambda session, message: debug_messages.append(message),
    )

    assert session.model_chunks == 1
    assert await session.queue.get() == "hello"
    assert debug_messages == ["model chunk #1 samples=1 sample_rate=16000"]


@pytest.mark.anyio
async def test_process_prepared_chunks_emits_inference_error() -> None:
    session = make_session()

    async def fake_inference(**kwargs):
        return type("Result", (), {"text": None, "error": "[server error] Live transcription failed: boom"})()

    await process_prepared_chunks(
        session=session,
        chunks=[np.array([0.1], dtype=np.float32)],
        target_sample_rate=16000,
        append_capture_audio_fn=lambda session, chunk: None,
        run_chunk_inference_fn=fake_inference,
        debug_fn=lambda session, message: None,
    )

    assert await session.queue.get() == "[server error] Live transcription failed: boom"


@pytest.mark.anyio
async def test_process_prepared_chunks_emits_recording_message_on_first_capture() -> None:
    session = make_session()

    async def fake_inference(**kwargs):
        return type("Result", (), {"text": None, "error": None})()

    capture_path = pathlib.Path("capture.wav")

    def fake_append_capture(session: SessionState, chunk: np.ndarray) -> pathlib.Path:
        session.capture_samples += int(chunk.size)
        return capture_path

    await process_prepared_chunks(
        session=session,
        chunks=[np.array([0.1, 0.2], dtype=np.float32)],
        target_sample_rate=16000,
        append_capture_audio_fn=fake_append_capture,
        run_chunk_inference_fn=fake_inference,
        debug_fn=lambda session, message: None,
    )

    assert await session.queue.get() == f"[server] recording WAV capture: {capture_path}"


@pytest.mark.anyio
async def test_process_prepared_chunks_limits_debug_to_first_four_chunks() -> None:
    session = make_session()
    debug_messages: list[str] = []

    async def fake_inference(**kwargs):
        return type("Result", (), {"text": None, "error": None})()

    await process_prepared_chunks(
        session=session,
        chunks=[np.array([0.1], dtype=np.float32) for _ in range(5)],
        target_sample_rate=16000,
        append_capture_audio_fn=lambda session, chunk: None,
        run_chunk_inference_fn=fake_inference,
        debug_fn=lambda session, message: debug_messages.append(message),
    )

    assert len(debug_messages) == 4
