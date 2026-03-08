from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TranscribeStreamChunkResponse:
    buffered_audio: np.ndarray
    stream_sample_rate: int | None
    model_inputs: list[np.ndarray] = field(default_factory=list)
    rejected_by_preprocessor: bool = False
    error: str | None = None
