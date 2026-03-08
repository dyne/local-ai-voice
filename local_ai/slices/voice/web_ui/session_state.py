from __future__ import annotations

import asyncio
import pathlib
import wave
from dataclasses import dataclass, field
from typing import Callable

from local_ai.slices.voice.web_ui.server_config import validate_chunk_config

DEFAULT_AUDIO_BITRATE = 48000


@dataclass
class SessionState:
    session_id: str
    queue: asyncio.Queue[str]
    save_sample: bool
    silence_detect: bool
    debug: bool
    audio_preprocessor: object | None
    chunk_seconds: float
    overlap_seconds: float
    stream_sample_rate: int | None = None
    mime_type: str | None = None
    audio_bitrate: int = DEFAULT_AUDIO_BITRATE
    capture_path: pathlib.Path | None = None
    capture_writer: wave.Wave_write | None = None
    capture_sample_rate: int | None = None
    capture_samples: int = 0
    audio_socket: object | None = None
    encoded_buffer: bytearray = field(default_factory=bytearray)
    decoded_sample_cursor: int = 0
    decoded_sample_rate: int | None = None
    received_messages: int = 0
    decode_attempts: int = 0
    decoded_messages: int = 0
    model_chunks: int = 0
    debug_messages_sent: int = 0


def create_session_state(
    *,
    session_id: str,
    save_sample: bool,
    silence_detect: bool,
    debug: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    mime_type: str | None,
    audio_bitrate: int,
    create_preprocessor: Callable[[bool, int], object | None],
    vad_mode: int,
) -> SessionState:
    validate_chunk_config(chunk_seconds, overlap_seconds)
    if audio_bitrate <= 0:
        raise ValueError("audio_bitrate must be > 0")
    try:
        audio_preprocessor = create_preprocessor(silence_detect, vad_mode)
    except Exception as exc:
        raise RuntimeError(f"Audio preprocessing failed: {exc}") from exc

    return SessionState(
        session_id=session_id,
        queue=asyncio.Queue(maxsize=64),
        save_sample=save_sample,
        silence_detect=silence_detect,
        debug=debug,
        audio_preprocessor=audio_preprocessor,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        mime_type=mime_type,
        audio_bitrate=audio_bitrate,
    )
