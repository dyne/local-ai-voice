from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscribeLiveRequest:
    chunk_seconds: float
    silence_detect: bool
    verbose: bool = False
