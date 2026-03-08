from __future__ import annotations

import numpy as np
from typing import Callable

from local_ai.slices.voice.shared.audio_processing import (
    TARGET_SAMPLE_RATE,
    preprocess_audio,
    resample_audio_linear,
)
from local_ai.slices.voice.transcribe_stream.request import TranscribeStreamChunkRequest
from local_ai.slices.voice.transcribe_stream.response import TranscribeStreamChunkResponse


def prepare_stream_chunks(
    *,
    request: TranscribeStreamChunkRequest,
    logger: Callable[[str, bool, float | None], None],
) -> TranscribeStreamChunkResponse:
    buffered = request.buffered_audio
    stream_sample_rate = request.current_stream_sample_rate

    if request.incoming_audio.size == 0:
        return TranscribeStreamChunkResponse(
            buffered_audio=buffered,
            stream_sample_rate=stream_sample_rate,
        )

    if stream_sample_rate is None:
        stream_sample_rate = request.incoming_sample_rate
    elif stream_sample_rate != request.incoming_sample_rate:
        if buffered.size > 0:
            buffered = resample_audio_linear(buffered, stream_sample_rate, request.incoming_sample_rate)
        stream_sample_rate = request.incoming_sample_rate

    buffered = np.concatenate((buffered, request.incoming_audio))
    chunk_samples = int(round(request.chunk_seconds * stream_sample_rate))
    stride_samples = chunk_samples - int(round(request.overlap_seconds * stream_sample_rate))
    if stride_samples <= 0:
        return TranscribeStreamChunkResponse(
            buffered_audio=buffered,
            stream_sample_rate=stream_sample_rate,
            error="Invalid chunk configuration.",
        )

    model_inputs: list[np.ndarray] = []
    rejected_by_preprocessor = False
    while buffered.shape[0] >= chunk_samples:
        chunk = buffered[:chunk_samples]
        buffered = buffered[stride_samples:]
        if np.max(np.abs(chunk), initial=0.0) < 1e-4:
            continue
        chunk = preprocess_audio(
            chunk,
            stream_sample_rate,
            request.audio_preprocessor,
            request.verbose,
            request.start,
            logger,
        )
        if request.silence_detect and chunk.size == 0:
            rejected_by_preprocessor = True
            continue
        if stream_sample_rate != TARGET_SAMPLE_RATE:
            chunk = resample_audio_linear(chunk, stream_sample_rate, TARGET_SAMPLE_RATE)
        model_inputs.append(chunk)

    return TranscribeStreamChunkResponse(
        buffered_audio=buffered,
        stream_sample_rate=stream_sample_rate,
        model_inputs=model_inputs,
        rejected_by_preprocessor=rejected_by_preprocessor,
    )
