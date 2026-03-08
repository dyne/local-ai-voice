from __future__ import annotations

import pathlib
import wave

import numpy as np
import pytest

from local_ai.slices.voice.shared import audio_processing


def create_test_wav(path: pathlib.Path, *, sample_rate: int = 8000, channels: int = 2) -> None:
    samples = np.array([0, 32767, -32768, 1000], dtype=np.int16)
    if channels == 2:
        samples = np.column_stack((samples, samples))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def test_read_wav_mono_float32_downmixes_to_mono(tmp_path: pathlib.Path) -> None:
    wav_path = tmp_path / "sample.wav"
    create_test_wav(wav_path)

    audio, sample_rate = audio_processing.read_wav_mono_float32(wav_path)

    assert sample_rate == 8000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 4


def test_normalize_audio_format_returns_flat_float32() -> None:
    normalized = audio_processing.normalize_audio_format(np.array([[1, 2], [3, 4]], dtype=np.int16))
    assert normalized.dtype == np.float32
    assert normalized.shape == (4,)


def test_resample_audio_linear_rejects_invalid_rates() -> None:
    with pytest.raises(ValueError, match="Invalid sample rate"):
        audio_processing.resample_audio_linear(np.array([0.0], dtype=np.float32), 0, 16000)


def test_ensure_sample_rate_logs_and_resamples() -> None:
    messages: list[str] = []
    audio = np.array([0.0, 1.0], dtype=np.float32)

    resampled, rate = audio_processing.ensure_sample_rate(
        audio,
        8000,
        True,
        1.0,
        lambda message, verbose, start: messages.append(message),
    )

    assert rate == 16000
    assert resampled.dtype == np.float32
    assert messages == ["Resampling audio from 8000 Hz to 16000 Hz"]


def test_create_audio_preprocessor_disabled_returns_none() -> None:
    assert audio_processing.create_audio_preprocessor(False) is None


def test_create_audio_preprocessor_rejects_invalid_params() -> None:
    with pytest.raises(RuntimeError, match="min_speech_frames must be > 0"):
        audio_processing.create_audio_preprocessor(True, min_speech_frames=0)


def test_prepare_vad_audio_uses_closest_supported_rate() -> None:
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    prepared, rate = audio_processing.prepare_vad_audio(audio, 11025)
    assert rate in audio_processing.SUPPORTED_VAD_SAMPLE_RATES
    assert prepared.dtype == np.float32


def test_preprocess_audio_without_preprocessor_normalizes_input() -> None:
    audio = audio_processing.preprocess_audio(
        np.array([[1, 2], [3, 4]], dtype=np.int16),
        16000,
        None,
        False,
        0.0,
        lambda message, verbose, start: None,
    )
    assert audio.dtype == np.float32
    assert audio.shape == (4,)
