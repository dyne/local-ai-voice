from __future__ import annotations

import io
import types

import numpy as np
import pytest

from local_ai.slices.voice.web_ui.audio_decode import decode_audio_frame, try_decode_bytes


class FakeFrame:
    def __init__(self, array: np.ndarray, sample_rate: int) -> None:
        self._array = array
        self.sample_rate = sample_rate

    def to_ndarray(self) -> np.ndarray:
        return self._array


def test_decode_audio_frame_downmixes_and_scales_integer_audio() -> None:
    frame = FakeFrame(np.array([[0, 32767], [0, 32767]], dtype=np.int16), 16000)

    audio = decode_audio_frame(frame)

    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 1.0], rel=1e-4)


def test_try_decode_bytes_concatenates_decoded_frames() -> None:
    frames = [
        FakeFrame(np.array([[0, 32767]], dtype=np.int16), 16000),
        FakeFrame(np.array([[32767, 0]], dtype=np.int16), 16000),
    ]

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def decode(self, audio: int):
            return iter(frames)

    result = try_decode_bytes(
        payload=b"abc",
        mime_type="audio/webm",
        open_container=lambda source, format=None: FakeContainer(),
    )

    assert result is not None
    audio, sample_rate = result
    assert sample_rate == 16000
    assert audio.tolist() == pytest.approx([0.0, 1.0, 1.0, 0.0], rel=1e-4)


def test_try_decode_bytes_returns_none_when_no_frames_decoded() -> None:
    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def decode(self, audio: int):
            return iter([])

    result = try_decode_bytes(
        payload=b"abc",
        mime_type="audio/ogg",
        open_container=lambda source, format=None: FakeContainer(),
    )

    assert result is None
