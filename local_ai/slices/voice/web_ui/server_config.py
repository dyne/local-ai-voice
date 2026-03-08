from __future__ import annotations

import pathlib


def validate_chunk_config(chunk_seconds: float, overlap_seconds: float) -> None:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")


def validate_tls_paths(certfile: pathlib.Path | None, keyfile: pathlib.Path | None) -> None:
    if (certfile is None) != (keyfile is None):
        raise ValueError("--tls-certfile and --tls-keyfile must be provided together.")
    if certfile is not None and not certfile.exists():
        raise ValueError(f"TLS certificate file not found: {certfile}")
    if keyfile is not None and not keyfile.exists():
        raise ValueError(f"TLS key file not found: {keyfile}")


def mime_type_to_av_format(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    lowered = mime_type.lower()
    if "ogg" in lowered:
        return "ogg"
    if "webm" in lowered:
        return "webm"
    return None


def desktop_host() -> str:
    return "127.0.0.1"
