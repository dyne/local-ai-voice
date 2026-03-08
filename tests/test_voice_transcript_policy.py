from __future__ import annotations

import types

import numpy as np
import pytest

from local_ai.slices.voice.shared.audio_processing import AudioPreprocessor
from local_ai.slices.voice.shared.transcript_policy import (
    result_to_text,
    setup_error_exit_code,
    should_suppress_transcript,
    transcribe_chunk,
)


def test_result_to_text_uses_text_attribute() -> None:
    result = types.SimpleNamespace(text="hello")
    assert result_to_text(result) == "hello"


def test_result_to_text_uses_first_texts_entry() -> None:
    result = types.SimpleNamespace(texts=["hello", "world"])
    assert result_to_text(result) == "hello"


def test_transcribe_chunk_passes_list_audio_to_pipe() -> None:
    class FakePipe:
        def __init__(self) -> None:
            self.calls: list[tuple[list[float], dict[str, object]]] = []

        def generate(self, audio: list[float], **kwargs: object) -> object:
            self.calls.append((audio, kwargs))
            return types.SimpleNamespace(text="  hi  ")

    pipe = FakePipe()
    text = transcribe_chunk(pipe, np.array([0.1, -0.2], dtype=np.float32), {"language": "<|en|>"})

    assert text == "hi"
    assert pipe.calls[0][0] == pytest.approx([0.1, -0.2])
    assert pipe.calls[0][1] == {"language": "<|en|>"}


def test_should_suppress_transcript_for_weak_hallucination() -> None:
    preprocessor = AudioPreprocessor(
        nr=object(),
        vad=object(),
        vad_mode=3,
        min_speech_frames=3,
        min_speech_ratio=0.2,
        min_utterance_ms=180,
        hangover_ms=300,
        last_speech_frames=2,
        last_total_frames=10,
        last_max_run=2,
        last_was_hangover=False,
    )

    assert should_suppress_transcript("uh", preprocessor) is True


def test_should_not_suppress_real_transcript() -> None:
    preprocessor = AudioPreprocessor(
        nr=object(),
        vad=object(),
        vad_mode=3,
        min_speech_frames=3,
        min_speech_ratio=0.2,
        min_utterance_ms=180,
        hangover_ms=300,
        last_speech_frames=2,
        last_total_frames=10,
        last_max_run=2,
        last_was_hangover=False,
    )

    assert should_suppress_transcript("hello world", preprocessor) is False


def test_setup_error_exit_code_maps_known_errors() -> None:
    assert setup_error_exit_code("Model directory not found: x") == 2
    assert setup_error_exit_code("Failed to create WhisperPipeline on GPU.") == 4
    assert setup_error_exit_code("anything else") == 3
