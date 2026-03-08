from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass(frozen=True)
class ChunkInferenceResult:
    text: str | None = None
    error: str | None = None


async def run_chunk_inference(
    *,
    chunk: object,
    pipe: object,
    generate_kwargs: dict[str, object],
    audio_preprocessor: object | None,
    infer_lock: object,
    transcribe_fn: Callable[[object, object, dict[str, object]], str],
    should_suppress_fn: Callable[[str, object | None], bool],
    likely_reason_details_fn: Callable[[Exception], list[str]],
    to_thread_fn: Callable[..., Awaitable[str]],
) -> ChunkInferenceResult:
    async with infer_lock:
        try:
            text = await to_thread_fn(transcribe_fn, pipe, chunk, generate_kwargs)
        except Exception as exc:
            details = likely_reason_details_fn(exc)
            return ChunkInferenceResult(error=f"[server error] Live transcription failed: {details[0]}")
    if text and not should_suppress_fn(text, audio_preprocessor):
        return ChunkInferenceResult(text=text)
    return ChunkInferenceResult()
