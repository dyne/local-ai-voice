from __future__ import annotations

from collections.abc import Awaitable, Callable

import numpy as np

from local_ai.slices.voice.web_ui.session_state import SessionState


async def process_prepared_chunks(
    *,
    session: SessionState,
    chunks: list[np.ndarray],
    target_sample_rate: int,
    append_capture_audio_fn: Callable[[SessionState, np.ndarray], object | None],
    run_chunk_inference_fn: Callable[..., Awaitable[object]],
    debug_fn: Callable[[SessionState, str], Awaitable[None] | object],
) -> None:
    for chunk in chunks:
        out_path = append_capture_audio_fn(session, chunk)
        if out_path is not None and session.capture_samples == int(chunk.size):
            await session.queue.put(f"[server] recording WAV capture: {out_path}")
        session.model_chunks += 1
        if session.model_chunks <= 4:
            maybe_awaitable = debug_fn(session, f"model chunk #{session.model_chunks} samples={chunk.size} sample_rate={target_sample_rate}")
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

        inference = await run_chunk_inference_fn(chunk=chunk, audio_preprocessor=session.audio_preprocessor)
        if getattr(inference, "error", None) is not None:
            await session.queue.put(inference.error)
            continue
        if getattr(inference, "text", None) is not None:
            await session.queue.put(inference.text)
