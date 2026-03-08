from __future__ import annotations

import pathlib
import wave
from dataclasses import dataclass
from typing import Callable

import numpy as np

TARGET_SAMPLE_RATE = 16000
SUPPORTED_VAD_SAMPLE_RATES = (8000, 16000, 32000, 48000)
VAD_FRAME_MS = 30
VAD_MODE = 3
VAD_MIN_SPEECH_FRAMES = 3
VAD_MIN_SPEECH_RATIO = 0.2
VAD_MIN_UTTERANCE_MS = 180
VAD_HANGOVER_MS = 300
NR_IMPORT_ERROR = "Install noisereduce and webrtcvad-wheels."


@dataclass
class AudioPreprocessor:
    nr: object
    vad: object
    vad_mode: int
    min_speech_frames: int
    min_speech_ratio: float
    min_utterance_ms: int
    hangover_ms: int
    hangover_frames: int = 0
    last_speech_frames: int = 0
    last_total_frames: int = 0
    last_max_run: int = 0
    last_was_hangover: bool = False


def read_wav_mono_float32(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def normalize_audio_format(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def resample_audio_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Invalid sample rate(s): source={source_rate}, target={target_rate}")
    if audio.size == 0:
        return audio

    duration = audio.shape[0] / float(source_rate)
    out_length = max(1, int(round(duration * target_rate)))
    in_times = np.arange(audio.shape[0], dtype=np.float64) / float(source_rate)
    out_times = np.arange(out_length, dtype=np.float64) / float(target_rate)
    return np.interp(out_times, in_times, audio).astype(np.float32)


def ensure_sample_rate(
    audio: np.ndarray,
    sample_rate: int,
    verbose: bool,
    start: float,
    logger: Callable[[str, bool, float | None], None],
) -> tuple[np.ndarray, int]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio, sample_rate
    logger(f"Resampling audio from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz", verbose, start)
    return resample_audio_linear(audio, sample_rate, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE


def create_audio_preprocessor(
    enabled: bool,
    *,
    vad_mode: int = VAD_MODE,
    min_speech_frames: int = VAD_MIN_SPEECH_FRAMES,
    min_speech_ratio: float = VAD_MIN_SPEECH_RATIO,
    min_utterance_ms: int = VAD_MIN_UTTERANCE_MS,
    hangover_ms: int = VAD_HANGOVER_MS,
) -> object | None:
    if not enabled:
        return None
    if vad_mode not in (0, 1, 2, 3):
        raise RuntimeError(f"Invalid VAD mode: {vad_mode}. Use 0, 1, 2, or 3.")
    if min_speech_frames <= 0:
        raise RuntimeError("min_speech_frames must be > 0.")
    if not (0.0 < min_speech_ratio <= 1.0):
        raise RuntimeError("min_speech_ratio must be > 0 and <= 1.")
    if min_utterance_ms <= 0:
        raise RuntimeError("min_utterance_ms must be > 0.")
    if hangover_ms < 0:
        raise RuntimeError("hangover_ms must be >= 0.")
    try:
        import noisereduce as nr
        import webrtcvad
    except Exception as exc:
        raise RuntimeError("Failed to import audio preprocessing dependencies.") from exc
    return AudioPreprocessor(
        nr=nr,
        vad=webrtcvad.Vad(vad_mode),
        vad_mode=vad_mode,
        min_speech_frames=min_speech_frames,
        min_speech_ratio=min_speech_ratio,
        min_utterance_ms=min_utterance_ms,
        hangover_ms=hangover_ms,
    )


def preferred_vad_sample_rate(sample_rate: int) -> int:
    return min(SUPPORTED_VAD_SAMPLE_RATES, key=lambda rate: abs(rate - sample_rate))


def prepare_vad_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate in SUPPORTED_VAD_SAMPLE_RATES:
        return normalize_audio_format(audio), sample_rate
    vad_sample_rate = preferred_vad_sample_rate(sample_rate)
    return resample_audio_linear(normalize_audio_format(audio), sample_rate, vad_sample_rate), vad_sample_rate


def speech_frame_stats(audio: np.ndarray, vad: object, sample_rate: int) -> tuple[int, int, int]:
    vad_audio, vad_sample_rate = prepare_vad_audio(audio, sample_rate)
    frame_samples = vad_sample_rate * VAD_FRAME_MS // 1000
    pcm16 = (np.clip(vad_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    if pcm16.size == 0:
        return 0, 0, 0
    speech_frames = 0
    max_consecutive = 0
    consecutive = 0
    total_frames = 0
    for start_idx in range(0, pcm16.shape[0], frame_samples):
        frame = pcm16[start_idx:start_idx + frame_samples]
        if frame.size == 0:
            continue
        if frame.size < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.size))
        total_frames += 1
        if vad.is_speech(frame.tobytes(), vad_sample_rate):
            speech_frames += 1
            consecutive += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0
    return speech_frames, total_frames, max_consecutive


def _speech_detected(
    original_audio: np.ndarray,
    reduced_audio: np.ndarray,
    preprocessor: AudioPreprocessor,
    sample_rate: int,
) -> bool:
    min_duration_frames = max(1, int(np.ceil(preprocessor.min_utterance_ms / float(VAD_FRAME_MS))))
    hangover_frames = max(1, int(np.ceil(preprocessor.hangover_ms / float(VAD_FRAME_MS))))

    orig_speech, orig_total, orig_run = speech_frame_stats(original_audio, preprocessor.vad, sample_rate)
    red_speech, red_total, red_run = speech_frame_stats(reduced_audio, preprocessor.vad, sample_rate)
    total_frames = max(orig_total, red_total)
    if total_frames == 0:
        preprocessor.hangover_frames = 0
        preprocessor.last_speech_frames = 0
        preprocessor.last_total_frames = 0
        preprocessor.last_max_run = 0
        preprocessor.last_was_hangover = False
        return False

    speech_frames = max(orig_speech, red_speech)
    consecutive_run = max(orig_run, red_run)
    preprocessor.last_speech_frames = speech_frames
    preprocessor.last_total_frames = total_frames
    preprocessor.last_max_run = consecutive_run
    preprocessor.last_was_hangover = False
    speech_ratio = speech_frames / float(total_frames)
    speech_now = (
        consecutive_run >= preprocessor.min_speech_frames
        and speech_frames >= min_duration_frames
        and speech_ratio >= preprocessor.min_speech_ratio
    )
    if speech_now:
        preprocessor.hangover_frames = hangover_frames
        return True
    if preprocessor.hangover_frames > 0:
        preprocessor.hangover_frames -= min(total_frames, preprocessor.hangover_frames)
        preprocessor.last_was_hangover = True
        return True
    return False


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    preprocessor: object | None,
    verbose: bool,
    start: float,
    logger: Callable[[str, bool, float | None], None],
) -> np.ndarray:
    if preprocessor is None:
        return normalize_audio_format(audio)
    original_audio = normalize_audio_format(audio)
    try:
        reduced = preprocessor.nr.reduce_noise(y=original_audio, sr=sample_rate)
    except Exception as exc:
        raise RuntimeError("Noise reduction failed.") from exc

    reduced_audio = normalize_audio_format(reduced)
    if reduced_audio.size == 0:
        return reduced_audio
    if not _speech_detected(original_audio, reduced_audio, preprocessor, sample_rate):
        logger("Audio skipped: no speech detected by VAD", verbose, start)
        return np.asarray([], dtype=np.float32)
    return reduced_audio
