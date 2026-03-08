from __future__ import annotations

import asyncio

import numpy as np
import pytest

from local_ai.slices.voice.web_ui.session_decoder import decode_session_message
from local_ai.slices.voice.web_ui.session_state import SessionState


def make_session() -> SessionState:
    return SessionState(
        session_id="abc",
        queue=asyncio.Queue(maxsize=1),
        save_sample=False,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


def test_decode_session_message_updates_state_and_returns_audio() -> None:
    session = make_session()

    audio, sample_rate = decode_session_message(
        session=session,
        message={"bytes": b"abc"},
        max_encoded_buffer_bytes=10,
        decode_message=lambda **kwargs: type(
            "Result",
            (),
            {
                "encoded_buffer": bytearray(b"abc"),
                "decoded_sample_cursor": 3,
                "decoded_sample_rate": 16000,
                "audio": np.array([1.0], dtype=np.float32),
                "sample_rate": 16000,
            },
        )(),
        invalid_data_error_type=RuntimeError,
    )

    assert session.decode_attempts == 1
    assert session.encoded_buffer == bytearray(b"abc")
    assert session.decoded_sample_cursor == 3
    assert session.decoded_sample_rate == 16000
    assert audio.tolist() == [1.0]
    assert sample_rate == 16000


def test_decode_session_message_returns_none_when_decoder_returns_no_audio() -> None:
    session = make_session()

    result = decode_session_message(
        session=session,
        message={"bytes": b"abc"},
        max_encoded_buffer_bytes=10,
        decode_message=lambda **kwargs: type(
            "Result",
            (),
            {
                "encoded_buffer": bytearray(b"abc"),
                "decoded_sample_cursor": 3,
                "decoded_sample_rate": 16000,
                "audio": None,
                "sample_rate": None,
            },
        )(),
        invalid_data_error_type=RuntimeError,
    )

    assert result is None


def test_decode_session_message_resets_state_on_runtime_error() -> None:
    session = make_session()
    session.encoded_buffer = bytearray(b"abc")
    session.decoded_sample_cursor = 2
    session.decoded_sample_rate = 16000

    with pytest.raises(RuntimeError, match="boom"):
        decode_session_message(
            session=session,
            message={"bytes": b"abc"},
            max_encoded_buffer_bytes=10,
            decode_message=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            invalid_data_error_type=RuntimeError,
        )

    assert session.encoded_buffer == bytearray()
    assert session.decoded_sample_cursor == 0
    assert session.decoded_sample_rate is None
