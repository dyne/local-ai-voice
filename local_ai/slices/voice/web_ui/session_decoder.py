from __future__ import annotations

from typing import Callable

import numpy as np

from local_ai.slices.voice.web_ui.session_state import SessionState


def decode_session_message(
    *,
    session: SessionState,
    message: dict[str, object],
    max_encoded_buffer_bytes: int,
    decode_message: Callable[..., object],
    invalid_data_error_type: type[Exception],
) -> tuple[np.ndarray, int] | None:
    raw = message.get("bytes")
    if raw:
        session.decode_attempts += 1
    try:
        decoded = decode_message(
            raw=raw if isinstance(raw, (bytes, bytearray)) else None,
            encoded_buffer=session.encoded_buffer,
            decoded_sample_cursor=session.decoded_sample_cursor,
            decoded_sample_rate=session.decoded_sample_rate,
            mime_type=session.mime_type,
            max_encoded_buffer_bytes=max_encoded_buffer_bytes,
            invalid_data_error_type=invalid_data_error_type,
        )
    except RuntimeError:
        session.encoded_buffer.clear()
        session.decoded_sample_cursor = 0
        session.decoded_sample_rate = None
        raise

    session.encoded_buffer = decoded.encoded_buffer
    session.decoded_sample_cursor = decoded.decoded_sample_cursor
    session.decoded_sample_rate = decoded.decoded_sample_rate
    if decoded.audio is None or decoded.sample_rate is None:
        return None
    return decoded.audio, decoded.sample_rate
