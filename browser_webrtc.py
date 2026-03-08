#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

from local_ai.slices.voice.web_ui.launch_helpers import fallback_message, find_free_port, wait_for_server
from local_ai.slices.voice.web_ui.launch_modes import run_desktop_mode, run_server_mode
from local_ai.slices.voice.web_ui.page_loader import load_index_html as load_browser_index_html
from local_ai.slices.voice.web_ui.runtime_context import create_server_context
from local_ai.slices.voice.web_ui.server_bootstrap import prepare_server_components
from local_ai.slices.voice.web_ui.server_config import (
    desktop_host,
    validate_chunk_config,
    validate_tls_paths,
)
from local_ai.slices.voice.web_ui.service import (
    AudioStreamService,
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAP_SECONDS,
    ServerContext,
    SessionConfig,
)
from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai_voice import (
    configure_openvino_runtime_env,
    log,
)
from local_ai.infrastructure.openvino.whisper import create_whisper_runtime, likely_reason_details

UI_PATH = pathlib.Path(__file__).resolve().parent / "web" / "index.html"


def create_context(args: argparse.Namespace, start_time: float) -> ServerContext:
    return create_server_context(
        args=args,
        start_time=start_time,
        configure_runtime_env=configure_openvino_runtime_env,
        create_runtime=create_whisper_runtime,
        logger=log,
        base_dir=pathlib.Path(__file__).resolve().parent,
        context_factory=ServerContext,
        lock_factory=asyncio.Lock,
    )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser page and transcribe client microphone over Opus/WebSocket.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server (default: 8000).")
    parser.add_argument("--tls-certfile", type=pathlib.Path, default=None, help="TLS certificate file (PEM).")
    parser.add_argument("--tls-keyfile", type=pathlib.Path, default=None, help="TLS private key file (PEM).")
    parser.add_argument("--device", default="NPU,GPU,CPU", help="Device preference order using NPU,GPU,CPU, or 'list' to print detected devices.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenVINO model directory or Hugging Face repo id. If omitted, default OpenVINO model is auto-downloaded.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable model downloads. Fail if required model is not available locally.",
    )
    parser.add_argument("--language", default=None, help="Optional language token like <|en|>.")
    parser.add_argument("--task", default=None, choices=["transcribe", "translate"], help="Optional Whisper task.")
    parser.add_argument("--timestamps", action="store_true", help="Request timestamps in result object.")
    silence_group = parser.add_mutually_exclusive_group()
    silence_group.add_argument(
        "--silence-detect",
        dest="silence_detect",
        action="store_true",
        help="Enable noise reduction and WebRTC VAD speech gating for browser sessions by default.",
    )
    silence_group.add_argument(
        "--no-silence-detect",
        dest="silence_detect",
        action="store_false",
        help="Disable noise reduction and WebRTC VAD speech gating by default; the browser checkbox can still enable it per session.",
    )
    parser.set_defaults(silence_detect=True)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=3, help="Default WebRTC VAD aggressiveness mode for browser sessions.")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS, help=f"Chunk duration in seconds for server-side transcription windows (default: {DEFAULT_CHUNK_SECONDS}).")
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS, help=f"Chunk overlap in seconds to preserve context across windows (default: {DEFAULT_OVERLAP_SECONDS}).")
    parser.add_argument("--profile", action="store_true", help="Enable py-spy profiling for this run.")
    parser.add_argument("--profile-output", type=pathlib.Path, default=None, help="Optional py-spy output SVG path (default: profiles/<timestamp>.svg).")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")
    return parser.parse_args(argv)


def prepare_server(args: argparse.Namespace) -> tuple[ServerContext, AudioStreamService, float]:
    ctx, service, start_time = prepare_server_components(
        args=args,
        perf_counter=time.perf_counter,
        validate_tls=validate_tls_paths,
        validate_chunk=validate_chunk_config,
        create_context_fn=create_context,
        load_index_html_fn=lambda silence_detect_default, vad_mode_default: load_browser_index_html(
            ui_path=UI_PATH,
            silence_detect_default=silence_detect_default,
            vad_mode_default=vad_mode_default,
        ),
        service_factory=lambda *, ctx, index_html: AudioStreamService(
            ctx=ctx,
            index_html=index_html,
            logger=log,
            likely_reason_details_fn=likely_reason_details,
        ),
    )
    return ctx, service, start_time


def run_server(args: argparse.Namespace) -> int:
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice-web",
        output_path=args.profile_output,
    )
    try:
        return run_server_mode(
            args=args,
            prepare_server_fn=prepare_server,
            enable_loopback_only_network_fn=enable_loopback_only_network,
            import_uvicorn_fn=lambda: __import__("uvicorn"),
            logger=log,
            stderr=sys.stderr,
        )
    finally:
        stop_py_spy_profile(profile_session)


def _print_fallback_url(args: argparse.Namespace) -> None:
    print(fallback_message(host=args.host, port=args.port, tls_certfile=args.tls_certfile), file=sys.stderr, flush=True)


def run_desktop(args: argparse.Namespace) -> int:
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice-webview",
        output_path=args.profile_output,
    )
    try:
        return run_desktop_mode(
            args=args,
            prepare_server_fn=prepare_server,
            run_server_fn=run_server,
            desktop_host_fn=desktop_host,
            find_free_port_fn=find_free_port,
            enable_loopback_only_network_fn=enable_loopback_only_network,
            import_desktop_dependencies_fn=lambda: (__import__("uvicorn"), __import__("webview")),
            wait_for_server_fn=wait_for_server,
            logger=log,
            print_fallback_url_fn=_print_fallback_url,
            thread_factory=__import__("threading").Thread,
            stderr=sys.stderr,
        )
    finally:
        stop_py_spy_profile(profile_session)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
