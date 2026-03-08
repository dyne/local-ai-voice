from __future__ import annotations

import numpy as np
import pytest

from local_ai.slices.voice.transcribe_stream.buffer_decoder import decode_audio_message


class FakeInvalidDataError(Exception):
    pass


def test_decode_audio_message_ignores_empty_payload() -> None:
    response = decode_audio_message(
        raw=None,
        encoded_buffer=bytearray(),
        decoded_sample_cursor=0,
        decoded_sample_rate=None,
        mime_type=None,
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: None,
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio is None
    assert response.sample_rate is None
    assert response.encoded_buffer == bytearray()


def test_decode_audio_message_waits_for_more_bytes_on_invalid_data() -> None:
    response = decode_audio_message(
        raw=b"abc",
        encoded_buffer=bytearray(),
        decoded_sample_cursor=0,
        decoded_sample_rate=None,
        mime_type="audio/webm",
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: (_ for _ in ()).throw(FakeInvalidDataError()),
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio is None
    assert response.sample_rate is None
    assert response.encoded_buffer == bytearray(b"abc")


def test_decode_audio_message_raises_on_buffer_overflow() -> None:
    with pytest.raises(RuntimeError, match="Encoded audio buffer overflow"):
        decode_audio_message(
            raw=b"123456",
            encoded_buffer=bytearray(b"12345"),
            decoded_sample_cursor=0,
            decoded_sample_rate=None,
            mime_type="audio/webm",
            max_encoded_buffer_bytes=8,
            decode_payload=lambda payload, mime_type: (_ for _ in ()).throw(FakeInvalidDataError()),
            invalid_data_error_type=FakeInvalidDataError,
        )


def test_decode_audio_message_returns_only_new_audio_after_cursor() -> None:
    response = decode_audio_message(
        raw=b"abc",
        encoded_buffer=bytearray(),
        decoded_sample_cursor=2,
        decoded_sample_rate=16000,
        mime_type="audio/webm",
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: (np.array([1.0, 2.0, 3.0], dtype=np.float32), 16000),
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio.tolist() == [3.0]
    assert response.sample_rate == 16000
    assert response.decoded_sample_cursor == 3


def test_decode_audio_message_resets_cursor_on_sample_rate_change() -> None:
    response = decode_audio_message(
        raw=b"abc",
        encoded_buffer=bytearray(),
        decoded_sample_cursor=2,
        decoded_sample_rate=8000,
        mime_type="audio/webm",
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: (np.array([1.0, 2.0], dtype=np.float32), 16000),
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio.tolist() == [1.0, 2.0]
    assert response.sample_rate == 16000
    assert response.decoded_sample_cursor == 2


def test_decode_audio_message_resets_cursor_when_cursor_exceeds_audio_size() -> None:
    response = decode_audio_message(
        raw=b"abc",
        encoded_buffer=bytearray(),
        decoded_sample_cursor=10,
        decoded_sample_rate=16000,
        mime_type="audio/webm",
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: (np.array([1.0, 2.0], dtype=np.float32), 16000),
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio.tolist() == [1.0, 2.0]
    assert response.decoded_sample_cursor == 2


def test_decode_audio_message_returns_none_when_no_new_audio() -> None:
    response = decode_audio_message(
        raw=b"abc",
        encoded_buffer=bytearray(),
        decoded_sample_cursor=2,
        decoded_sample_rate=16000,
        mime_type="audio/webm",
        max_encoded_buffer_bytes=10,
        decode_payload=lambda payload, mime_type: (np.array([1.0, 2.0], dtype=np.float32), 16000),
        invalid_data_error_type=FakeInvalidDataError,
    )

    assert response.audio is None
    assert response.sample_rate is None
