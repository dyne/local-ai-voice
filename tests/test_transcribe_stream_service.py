from __future__ import annotations

import numpy as np
import pytest

from local_ai.slices.voice.transcribe_stream.request import TranscribeStreamChunkRequest
from local_ai.slices.voice.transcribe_stream.service import prepare_stream_chunks


def test_prepare_stream_chunks_initializes_stream_rate() -> None:
    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.1, 0.2], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert response.stream_sample_rate == 16000
    assert response.buffered_audio.tolist() == pytest.approx([0.1, 0.2])
    assert response.model_inputs == []
    assert response.error is None


def test_prepare_stream_chunks_resamples_buffer_when_sample_rate_changes(monkeypatch) -> None:
    calls: list[tuple[list[float], int, int]] = []

    def fake_resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
        calls.append((audio.tolist(), src, dst))
        return np.array([9.0], dtype=np.float32)

    monkeypatch.setattr("local_ai.slices.voice.transcribe_stream.service.resample_audio_linear", fake_resample)

    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.5], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.array([1.0, 2.0], dtype=np.float32),
        current_stream_sample_rate=8000,
        chunk_seconds=10.0,
        overlap_seconds=0.0,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert calls == [([1.0, 2.0], 8000, 16000)]
    assert response.buffered_audio.tolist() == [9.0, 0.5]
    assert response.stream_sample_rate == 16000


def test_prepare_stream_chunks_returns_error_for_invalid_stride() -> None:
    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.1, 0.2], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=1.0,
        overlap_seconds=1.0,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert response.error == "Invalid chunk configuration."


def test_prepare_stream_chunks_emits_model_input_after_preprocess(monkeypatch) -> None:
    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_stream.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: audio,
    )

    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=0.00025,
        overlap_seconds=0.0,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert len(response.model_inputs) == 1
    assert response.model_inputs[0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert response.buffered_audio.size == 0


def test_prepare_stream_chunks_skips_preprocessor_rejected_chunk(monkeypatch) -> None:
    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_stream.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: np.asarray([], dtype=np.float32),
    )

    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=0.00025,
        overlap_seconds=0.0,
        silence_detect=True,
        audio_preprocessor=object(),
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert response.model_inputs == []
    assert response.rejected_by_preprocessor is True


def test_prepare_stream_chunks_resamples_model_input_to_target_rate(monkeypatch) -> None:
    def fake_preprocess(audio, sample_rate, preprocessor, verbose, start, logger):
        return audio

    calls: list[tuple[list[float], int, int]] = []

    def fake_resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
        calls.append((audio.tolist(), src, dst))
        return np.array([7.0, 8.0], dtype=np.float32)

    monkeypatch.setattr("local_ai.slices.voice.transcribe_stream.service.preprocess_audio", fake_preprocess)
    monkeypatch.setattr("local_ai.slices.voice.transcribe_stream.service.resample_audio_linear", fake_resample)

    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([0.1, 0.2], dtype=np.float32),
        incoming_sample_rate=8000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=0.00025,
        overlap_seconds=0.0,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert len(calls) == 1
    assert calls[0][0] == pytest.approx([0.1, 0.2])
    assert calls[0][1:] == (8000, 16000)
    assert response.model_inputs[0].tolist() == [7.0, 8.0]


def test_prepare_stream_chunks_preserves_overlap_buffer(monkeypatch) -> None:
    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_stream.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: audio,
    )

    request = TranscribeStreamChunkRequest(
        incoming_audio=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        incoming_sample_rate=16000,
        buffered_audio=np.asarray([], dtype=np.float32),
        current_stream_sample_rate=None,
        chunk_seconds=0.00025,
        overlap_seconds=0.000125,
        silence_detect=True,
        audio_preprocessor=None,
        verbose=False,
        start=0.0,
    )

    response = prepare_stream_chunks(request=request, logger=lambda message, verbose, start: None)

    assert len(response.model_inputs) == 1
    assert response.buffered_audio.tolist() == [3.0, 4.0]
