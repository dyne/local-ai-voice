from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscribeFileRequest:
    wav_path: pathlib.Path
    verbose: bool = False
