from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from local_ai.slices.voice.transcribe_stream.request import TranscribeStreamChunkRequest
from local_ai.slices.voice.web_ui.session_state import SessionState


@dataclass(frozen=True)
class MessageProcessingResult:
    buffered_audio: np.ndarray
    stop: bool = False


async def process_audio_message(
    *,
    session: SessionState,
    message: dict[str, object],
    buffered_audio: np.ndarray,
    verbose: bool,
    start_time: float,
    logger: Callable[[str, bool, float | None], None],
    debug_fn: Callable[[SessionState, str], Awaitable[None] | object],
    decode_audio_message_fn: Callable[[SessionState, dict[str, object]], tuple[np.ndarray, int] | None],
    prepare_stream_chunks_fn: Callable[..., object],
    process_prepared_chunks_fn: Callable[..., Awaitable[None] | object],
    cleanup_session_fn: Callable[[SessionState], Awaitable[None]],
) -> MessageProcessingResult:
    decoded = decode_audio_message_fn(session, message)
    if decoded is None:
        maybe_awaitable = debug_fn(session, f"blob #{session.received_messages} not decodable yet buffer={len(session.encoded_buffer)}")
        if hasattr(maybe_awaitable, "__await__"):
            await maybe_awaitable
        return MessageProcessingResult(buffered_audio=buffered_audio)

    audio, sample_rate = decoded
    session.decoded_messages += 1
    if audio.size == 0:
        return MessageProcessingResult(buffered_audio=buffered_audio)

    prepared = prepare_stream_chunks_fn(
        request=TranscribeStreamChunkRequest(
            incoming_audio=audio,
            incoming_sample_rate=sample_rate,
            buffered_audio=buffered_audio,
            current_stream_sample_rate=session.stream_sample_rate,
            chunk_seconds=session.chunk_seconds,
            overlap_seconds=session.overlap_seconds,
            silence_detect=session.silence_detect,
            audio_preprocessor=session.audio_preprocessor,
            verbose=verbose,
            start=start_time,
        ),
        logger=logger,
    )
    session.stream_sample_rate = prepared.stream_sample_rate
    if prepared.error is not None:
        await session.queue.put("[server error] Invalid chunk configuration.")
        await cleanup_session_fn(session)
        return MessageProcessingResult(buffered_audio=prepared.buffered_audio, stop=True)

    if prepared.rejected_by_preprocessor and session.model_chunks == 0:
        maybe_awaitable = debug_fn(session, "chunk rejected by preprocessing/VAD before model input")
        if hasattr(maybe_awaitable, "__await__"):
            await maybe_awaitable

    maybe_awaitable = process_prepared_chunks_fn(
        session=session,
        chunks=prepared.model_inputs,
    )
    if hasattr(maybe_awaitable, "__await__"):
        await maybe_awaitable
    return MessageProcessingResult(buffered_audio=prepared.buffered_audio)
