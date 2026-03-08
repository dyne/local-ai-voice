from __future__ import annotations

import io
from typing import Callable

import av
import numpy as np

from local_ai.slices.voice.shared.audio_processing import normalize_audio_format
from local_ai.slices.voice.web_ui.server_config import mime_type_to_av_format


def decode_audio_frame(frame: av.AudioFrame) -> np.ndarray:
    array = frame.to_ndarray()
    if array.ndim == 2:
        audio = array.mean(axis=0)
    else:
        audio = array
    audio = normalize_audio_format(audio)
    if np.issubdtype(array.dtype, np.integer):
        max_value = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        audio = audio / float(max_value)
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def try_decode_bytes(
    *,
    payload: bytes,
    mime_type: str | None,
    open_container: Callable[..., object] = av.open,
) -> tuple[np.ndarray, int] | None:
    format_hint = mime_type_to_av_format(mime_type)
    with open_container(io.BytesIO(payload), format=format_hint) as container:
        decoded_frames: list[np.ndarray] = []
        sample_rate: int | None = None
        for frame in container.decode(audio=0):
            mono = decode_audio_frame(frame)
            if mono.size == 0:
                continue
            decoded_frames.append(mono)
            sample_rate = int(frame.sample_rate)
    if not decoded_frames or sample_rate is None:
        return None
    return np.concatenate(decoded_frames), sample_rate
