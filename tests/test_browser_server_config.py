from __future__ import annotations

import pathlib

import pytest

from local_ai.slices.voice.web_ui.server_config import (
    desktop_host,
    mime_type_to_av_format,
    validate_chunk_config,
    validate_tls_paths,
)


def test_validate_chunk_config_rejects_non_positive_chunk_seconds() -> None:
    with pytest.raises(ValueError, match="chunk_seconds must be > 0"):
        validate_chunk_config(0.0, 0.0)


def test_validate_chunk_config_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_seconds must be >= 0"):
        validate_chunk_config(1.0, -0.1)


def test_validate_chunk_config_rejects_overlap_not_smaller_than_chunk() -> None:
    with pytest.raises(ValueError, match="overlap_seconds must be smaller than chunk_seconds"):
        validate_chunk_config(1.0, 1.0)


def test_validate_chunk_config_accepts_valid_values() -> None:
    validate_chunk_config(1.0, 0.25)


def test_validate_tls_paths_requires_both_paths(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="must be provided together"):
        validate_tls_paths(cert, None)


def test_validate_tls_paths_requires_existing_files(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="TLS key file not found"):
        validate_tls_paths(cert, key)


def test_validate_tls_paths_accepts_existing_files(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("x", encoding="utf-8")
    key.write_text("x", encoding="utf-8")

    validate_tls_paths(cert, key)


def test_mime_type_to_av_format_recognizes_supported_types() -> None:
    assert mime_type_to_av_format("audio/ogg; codecs=opus") == "ogg"
    assert mime_type_to_av_format("audio/webm") == "webm"
    assert mime_type_to_av_format("audio/wav") is None


def test_desktop_host_is_loopback() -> None:
    assert desktop_host() == "127.0.0.1"
