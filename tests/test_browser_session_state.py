from __future__ import annotations

import pytest

from local_ai.slices.voice.web_ui.session_state import create_session_state


def test_create_session_state_builds_session() -> None:
    session = create_session_state(
        session_id="abc",
        save_sample=True,
        silence_detect=True,
        debug=False,
        chunk_seconds=1.5,
        overlap_seconds=0.0,
        mime_type="audio/webm",
        audio_bitrate=48000,
        create_preprocessor=lambda enabled, vad_mode: {"enabled": enabled, "vad_mode": vad_mode},
        vad_mode=3,
    )

    assert session.session_id == "abc"
    assert session.mime_type == "audio/webm"
    assert session.audio_bitrate == 48000
    assert session.audio_preprocessor == {"enabled": True, "vad_mode": 3}


def test_create_session_state_rejects_invalid_chunk_config() -> None:
    with pytest.raises(ValueError, match="chunk_seconds must be > 0"):
        create_session_state(
            session_id="abc",
            save_sample=False,
            silence_detect=True,
            debug=False,
            chunk_seconds=0.0,
            overlap_seconds=0.0,
            mime_type=None,
            audio_bitrate=48000,
            create_preprocessor=lambda enabled, vad_mode: None,
            vad_mode=3,
        )


def test_create_session_state_rejects_invalid_audio_bitrate() -> None:
    with pytest.raises(ValueError, match="audio_bitrate must be > 0"):
        create_session_state(
            session_id="abc",
            save_sample=False,
            silence_detect=True,
            debug=False,
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            mime_type=None,
            audio_bitrate=0,
            create_preprocessor=lambda enabled, vad_mode: None,
            vad_mode=3,
        )


def test_create_session_state_wraps_preprocessor_error() -> None:
    def raise_preprocessor(enabled, vad_mode):
        raise RuntimeError("no deps")

    with pytest.raises(RuntimeError, match="Audio preprocessing failed: no deps"):
        create_session_state(
            session_id="abc",
            save_sample=False,
            silence_detect=True,
            debug=False,
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            mime_type=None,
            audio_bitrate=48000,
            create_preprocessor=raise_preprocessor,
            vad_mode=3,
        )
