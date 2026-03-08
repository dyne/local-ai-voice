from __future__ import annotations

import asyncio

import numpy as np
import pytest

from local_ai.slices.voice.web_ui.message_processor import process_audio_message
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
async def test_process_audio_message_returns_unchanged_buffer_when_decode_not_ready() -> None:
    session = make_session()
    debug_messages: list[str] = []

    result = await process_audio_message(
        session=session,
        message={"bytes": b"abc"},
        buffered_audio=np.asarray([], dtype=np.float32),
        verbose=False,
        start_time=0.0,
        logger=lambda message, verbose, start: None,
        debug_fn=lambda session, message: debug_messages.append(message),
        decode_audio_message_fn=lambda session, message: None,
        prepare_stream_chunks_fn=lambda **kwargs: None,
        process_prepared_chunks_fn=lambda **kwargs: None,
        cleanup_session_fn=lambda session: None,
    )

    assert result.buffered_audio.size == 0
    assert debug_messages == ["blob #0 not decodable yet buffer=0"]


@pytest.mark.anyio
async def test_process_audio_message_returns_cleanup_on_invalid_chunk_configuration() -> None:
    session = make_session()
    cleaned: list[str] = []

    async def cleanup(session: SessionState) -> None:
        cleaned.append(session.session_id)

    result = await process_audio_message(
        session=session,
        message={"bytes": b"abc"},
        buffered_audio=np.asarray([], dtype=np.float32),
        verbose=False,
        start_time=0.0,
        logger=lambda message, verbose, start: None,
        debug_fn=lambda session, message: None,
        decode_audio_message_fn=lambda session, message: (np.array([0.1], dtype=np.float32), 16000),
        prepare_stream_chunks_fn=lambda **kwargs: type(
            "Prepared",
            (),
            {
                "buffered_audio": np.asarray([], dtype=np.float32),
                "stream_sample_rate": 16000,
                "error": "Invalid chunk configuration.",
                "rejected_by_preprocessor": False,
                "model_inputs": [],
            },
        )(),
        process_prepared_chunks_fn=lambda **kwargs: None,
        cleanup_session_fn=cleanup,
    )

    assert cleaned == ["abc"]
    assert await session.queue.get() == "[server error] Invalid chunk configuration."
    assert result.stop is True


@pytest.mark.anyio
async def test_process_audio_message_processes_prepared_chunks() -> None:
    session = make_session()
    processed: list[int] = []
    debug_messages: list[str] = []

    async def process_chunks(**kwargs) -> None:
        processed.append(len(kwargs["chunks"]))

    result = await process_audio_message(
        session=session,
        message={"bytes": b"abc"},
        buffered_audio=np.asarray([], dtype=np.float32),
        verbose=False,
        start_time=0.0,
        logger=lambda message, verbose, start: None,
        debug_fn=lambda session, message: debug_messages.append(message),
        decode_audio_message_fn=lambda session, message: (np.array([0.1], dtype=np.float32), 16000),
        prepare_stream_chunks_fn=lambda **kwargs: type(
            "Prepared",
            (),
            {
                "buffered_audio": np.array([0.2], dtype=np.float32),
                "stream_sample_rate": 16000,
                "error": None,
                "rejected_by_preprocessor": True,
                "model_inputs": [np.array([0.1], dtype=np.float32)],
            },
        )(),
        process_prepared_chunks_fn=process_chunks,
        cleanup_session_fn=lambda session: None,
    )

    assert result.buffered_audio.tolist() == pytest.approx([0.2])
    assert result.stop is False
    assert processed == [1]
    assert debug_messages == ["chunk rejected by preprocessing/VAD before model input"]
