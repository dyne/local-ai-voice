from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DecodedAudioMessage:
    encoded_buffer: bytearray
    decoded_sample_cursor: int
    decoded_sample_rate: int | None
    audio: np.ndarray | None
    sample_rate: int | None


def decode_audio_message(
    *,
    raw: bytes | None,
    encoded_buffer: bytearray,
    decoded_sample_cursor: int,
    decoded_sample_rate: int | None,
    mime_type: str | None,
    max_encoded_buffer_bytes: int,
    decode_payload: Callable[[bytes, str | None], tuple[np.ndarray, int] | None],
    invalid_data_error_type: type[Exception],
) -> DecodedAudioMessage:
    if not raw:
        return DecodedAudioMessage(
            encoded_buffer=encoded_buffer,
            decoded_sample_cursor=decoded_sample_cursor,
            decoded_sample_rate=decoded_sample_rate,
            audio=None,
            sample_rate=None,
        )

    updated_buffer = bytearray(encoded_buffer)
    updated_buffer.extend(raw)
    try:
        decoded = decode_payload(bytes(updated_buffer), mime_type)
    except invalid_data_error_type:
        if len(updated_buffer) > max_encoded_buffer_bytes:
            raise RuntimeError("Encoded audio buffer overflow; browser stream is not decodable.")
        return DecodedAudioMessage(
            encoded_buffer=updated_buffer,
            decoded_sample_cursor=decoded_sample_cursor,
            decoded_sample_rate=decoded_sample_rate,
            audio=None,
            sample_rate=None,
        )

    if decoded is None:
        return DecodedAudioMessage(
            encoded_buffer=updated_buffer,
            decoded_sample_cursor=decoded_sample_cursor,
            decoded_sample_rate=decoded_sample_rate,
            audio=None,
            sample_rate=None,
        )

    audio, sample_rate = decoded
    cursor = decoded_sample_cursor
    if decoded_sample_rate is not None and decoded_sample_rate != sample_rate:
        cursor = 0
    if cursor > audio.size:
        cursor = 0

    new_audio = audio[cursor:]
    cursor = int(audio.size)
    if new_audio.size == 0:
        return DecodedAudioMessage(
            encoded_buffer=updated_buffer,
            decoded_sample_cursor=cursor,
            decoded_sample_rate=sample_rate,
            audio=None,
            sample_rate=None,
        )

    return DecodedAudioMessage(
        encoded_buffer=updated_buffer,
        decoded_sample_cursor=cursor,
        decoded_sample_rate=sample_rate,
        audio=new_audio,
        sample_rate=sample_rate,
    )
