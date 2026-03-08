from __future__ import annotations

import pathlib
import time
import wave

import numpy as np

from local_ai.slices.voice.shared.audio_processing import TARGET_SAMPLE_RATE
from local_ai.slices.voice.web_ui.session_state import SessionState


def ensure_capture_writer(session: SessionState) -> pathlib.Path:
    if session.capture_writer is None:
        captures_dir = pathlib.Path.cwd() / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
        session.capture_path = captures_dir / f"capture_{session.session_id}_{int(time.time())}.wav"
        writer = wave.open(str(session.capture_path), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TARGET_SAMPLE_RATE)
        session.capture_writer = writer
        session.capture_sample_rate = TARGET_SAMPLE_RATE
    return session.capture_path


def append_capture_audio(session: SessionState, audio: np.ndarray) -> pathlib.Path | None:
    if not session.save_sample or audio.size == 0:
        return None
    out_path = ensure_capture_writer(session)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    session.capture_writer.writeframes(pcm16.tobytes())
    session.capture_samples += int(audio.size)
    return out_path


def close_capture_writer(session: SessionState) -> pathlib.Path | None:
    if session.capture_writer is None:
        return None
    try:
        session.capture_writer.close()
    finally:
        session.capture_writer = None
    return session.capture_path
