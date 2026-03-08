from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TranscribeStreamChunkRequest:
    incoming_audio: np.ndarray
    incoming_sample_rate: int
    buffered_audio: np.ndarray
    current_stream_sample_rate: int | None
    chunk_seconds: float
    overlap_seconds: float
    silence_detect: bool
    audio_preprocessor: object | None
    verbose: bool
    start: float
