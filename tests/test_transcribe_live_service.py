from __future__ import annotations

import pathlib
import types

import numpy as np

from local_ai.slices.voice.transcribe_live.request import TranscribeLiveRequest
from local_ai.slices.voice.transcribe_live.service import execute_transcribe_live


class FakeInputStream:
    def __init__(self, chunks, *, raise_on_enter: Exception | None = None) -> None:
        self._chunks = list(chunks)
        self._raise_on_enter = raise_on_enter

    def __enter__(self):
        if self._raise_on_enter is not None:
            raise self._raise_on_enter
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, chunk_samples: int):
        if not self._chunks:
            raise KeyboardInterrupt()
        item = self._chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_sounddevice(default_samplerate: float, chunks, *, query_error: Exception | None = None, stream_error: Exception | None = None):
    def query_devices(kind: str):
        if query_error is not None:
            raise query_error
        return {"default_samplerate": default_samplerate}

    def input_stream(**kwargs):
        return FakeInputStream(chunks, raise_on_enter=stream_error)

    return types.SimpleNamespace(query_devices=query_devices, InputStream=input_stream)


def test_execute_transcribe_live_rejects_non_positive_chunk_seconds() -> None:
    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=0.0, silence_detect=True, verbose=False),
        sounddevice_module=None,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 2
    assert response.reason == "--chunk-seconds must be > 0."


def test_execute_transcribe_live_reports_missing_sounddevice() -> None:
    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=ImportError("missing"),
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 6
    assert response.reason == "Live mode requires sounddevice."


def test_execute_transcribe_live_reports_default_sample_rate_query_error() -> None:
    sd = make_sounddevice(16000.0, [], query_error=RuntimeError("bad mic"))

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 7
    assert response.reason == "Could not read default input sample rate."


def test_execute_transcribe_live_reports_invalid_default_sample_rate() -> None:
    sd = make_sounddevice(0.0, [])

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 7
    assert response.reason == "Default input sample rate is invalid."


def test_execute_transcribe_live_emits_chunk_output(monkeypatch) -> None:
    chunks = [
        (np.array([[0.5], [0.5]], dtype=np.float32), False),
    ]
    sd = make_sounddevice(16000.0, chunks)
    outputs: list[str] = []
    statuses: list[str] = []

    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_live.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: audio,
    )
    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_live.service.transcribe_chunk",
        lambda pipe, audio, kwargs: "hello",
    )
    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_live.service.should_suppress_transcript",
        lambda text, preprocessor: False,
    )

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=True),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=outputs.append,
        on_status=statuses.append,
    )

    assert response.exit_code == 0
    assert outputs == ["[chunk 1] hello"]
    assert statuses[0] == "Live transcription started. Press Ctrl+C to stop."
    assert statuses[-1] == "\nStopped live transcription."


def test_execute_transcribe_live_skips_silent_audio(monkeypatch) -> None:
    chunks = [
        (np.array([[0.0], [0.0]], dtype=np.float32), False),
    ]
    sd = make_sounddevice(16000.0, chunks)
    outputs: list[str] = []

    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_live.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: audio,
    )

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=outputs.append,
        on_status=lambda line: None,
    )

    assert response.exit_code == 0
    assert outputs == []


def test_execute_transcribe_live_reports_preprocessing_error(monkeypatch) -> None:
    chunks = [
        (np.array([[0.5], [0.5]], dtype=np.float32), False),
    ]
    sd = make_sounddevice(16000.0, chunks)

    def raise_preprocess(audio, sample_rate, preprocessor, verbose, start, logger):
        raise RuntimeError("bad preprocess")

    monkeypatch.setattr("local_ai.slices.voice.transcribe_live.service.preprocess_audio", raise_preprocess)

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 6
    assert response.reason == "Live audio preprocessing failed."


def test_execute_transcribe_live_reports_inference_error(monkeypatch) -> None:
    chunks = [
        (np.array([[0.5], [0.5]], dtype=np.float32), False),
    ]
    sd = make_sounddevice(16000.0, chunks)

    monkeypatch.setattr(
        "local_ai.slices.voice.transcribe_live.service.preprocess_audio",
        lambda audio, sample_rate, preprocessor, verbose, start, logger: audio,
    )

    def raise_transcribe(pipe, audio, kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("local_ai.slices.voice.transcribe_live.service.transcribe_chunk", raise_transcribe)

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [f"Runtime error: {exc}"],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 5
    assert response.reason == "Live transcription failed."


def test_execute_transcribe_live_reports_capture_failure() -> None:
    sd = make_sounddevice(16000.0, [], stream_error=RuntimeError("mic busy"))

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(chunk_seconds=1.0, silence_detect=True, verbose=False),
        sounddevice_module=sd,
        pipe=object(),
        audio_preprocessor=None,
        generate_kwargs={},
        start=0.0,
        logger=lambda message, verbose, start: None,
        runtime_error_details=lambda exc: [str(exc)],
        on_output=lambda line: None,
        on_status=lambda line: None,
    )

    assert response.exit_code == 7
    assert response.reason == "Failed to capture microphone audio."
