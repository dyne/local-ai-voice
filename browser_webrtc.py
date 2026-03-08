#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import av
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from local_ai.slices.voice.web_ui.app_factory import build_browser_app
from local_ai.slices.voice.web_ui.chunk_pipeline import process_prepared_chunks
from local_ai.slices.voice.shared.audio_processing import (
    TARGET_SAMPLE_RATE,
    create_audio_preprocessor,
)
from local_ai.slices.voice.transcribe_stream.buffer_decoder import decode_audio_message
from local_ai.slices.voice.shared.transcript_policy import should_suppress_transcript, transcribe_chunk
from local_ai.slices.voice.transcribe_stream.request import TranscribeStreamChunkRequest
from local_ai.slices.voice.transcribe_stream.service import prepare_stream_chunks
from local_ai.slices.voice.web_ui.audio_decode import try_decode_bytes
from local_ai.slices.voice.web_ui.capture_store import (
    append_capture_audio,
)
from local_ai.slices.voice.web_ui.session_cleanup import cleanup_session
from local_ai.slices.voice.web_ui.event_stream import event_stream
from local_ai.slices.voice.web_ui.inference_runner import run_chunk_inference
from local_ai.slices.voice.web_ui.launch_helpers import fallback_message, find_free_port, wait_for_server
from local_ai.slices.voice.web_ui.launch_modes import run_desktop_mode, run_server_mode
from local_ai.slices.voice.web_ui.message_processor import process_audio_message
from local_ai.slices.voice.web_ui.runtime_context import create_server_context
from local_ai.slices.voice.web_ui.session_decoder import decode_session_message
from local_ai.slices.voice.web_ui.session_registry import (
    close_unknown_session,
    replace_existing_session,
    reset_existing_audio_socket_session,
)
from local_ai.slices.voice.web_ui.server_bootstrap import prepare_server_components
from local_ai.slices.voice.web_ui.server_config import (
    desktop_host,
    validate_chunk_config,
    validate_tls_paths,
)
from local_ai.slices.voice.web_ui.session_state import (
    DEFAULT_AUDIO_BITRATE,
    SessionState,
    create_session_state,
)
from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai_voice import (
    configure_openvino_runtime_env,
    log,
)
from local_ai.infrastructure.openvino.whisper import create_whisper_runtime, likely_reason_details

DEFAULT_CHUNK_SECONDS = 1.5
DEFAULT_OVERLAP_SECONDS = 0.0
MAX_ENCODED_BUFFER_BYTES = 4 * 1024 * 1024
UI_PATH = pathlib.Path(__file__).resolve().parent / "web" / "index.html"


class SessionConfig(BaseModel):
    session_id: str
    save_sample: bool = False
    silence_detect: bool = True
    debug: bool = False
    vad_mode: int = 3
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS
    mime_type: str | None = None
    audio_bitrate: int = DEFAULT_AUDIO_BITRATE


@dataclass
class ServerContext:
    pipe: object
    generate_kwargs: dict[str, object]
    selected_device: str
    model_dir: pathlib.Path
    silence_detect_default: bool
    vad_mode_default: int
    chunk_seconds: float
    overlap_seconds: float
    verbose: bool
    start_time: float
    infer_lock: asyncio.Lock


def load_index_html(silence_detect_default: bool, vad_mode_default: int) -> str:
    if UI_PATH.exists():
        checked_attr = "checked" if silence_detect_default else ""
        return (
            UI_PATH.read_text(encoding="utf-8")
            .replace("__SILENCE_DETECT_DEFAULT__", checked_attr)
            .replace("__VAD_MODE_DEFAULT__", str(vad_mode_default))
        )
    return "<!doctype html><html><body><h3>UI file missing</h3></body></html>"


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


class AudioStreamService:
    def __init__(self, ctx: ServerContext, index_html: str) -> None:
        self.ctx = ctx
        self.index_html = index_html
        self.sessions: dict[str, SessionState] = {}

    async def _debug(self, session: SessionState, message: str, limit: int = 12) -> None:
        if not session.debug:
            return
        if session.debug_messages_sent >= limit:
            return
        session.debug_messages_sent += 1
        await session.queue.put(f"[debug] {message}")

    def build_app(self) -> FastAPI:
        async def events(session_id: str) -> StreamingResponse:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Unknown session")
            return StreamingResponse(self._event_stream(session), media_type="text/event-stream")

        async def close_session(session_id: str) -> JSONResponse:
            session = self.sessions.get(session_id)
            if session is not None:
                await self._cleanup_session(session)
            return JSONResponse({"ok": True})

        return build_browser_app(
            index_html=self.index_html,
            create_session_handler=self._create_session,
            audio_handler=self._handle_audio_socket,
            events_handler=events,
            close_session_handler=close_session,
        )

    async def _create_session(self, payload: SessionConfig) -> JSONResponse:
        await replace_existing_session(payload.session_id, self.sessions, self._cleanup_session)

        try:
            self.sessions[payload.session_id] = create_session_state(
                session_id=payload.session_id,
                save_sample=payload.save_sample,
                silence_detect=payload.silence_detect,
                debug=payload.debug,
                chunk_seconds=payload.chunk_seconds,
                overlap_seconds=payload.overlap_seconds,
                mime_type=payload.mime_type,
                audio_bitrate=payload.audio_bitrate,
                create_preprocessor=lambda enabled, vad_mode: create_audio_preprocessor(enabled, vad_mode=vad_mode),
                vad_mode=payload.vad_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        await self._debug(
            self.sessions[payload.session_id],
            f"session created mime={payload.mime_type or 'unknown'} bitrate={payload.audio_bitrate} chunk={payload.chunk_seconds:.2f}s overlap={payload.overlap_seconds:.2f}s save_sample={payload.save_sample}",
        )
        return JSONResponse({"ok": True})

    async def _handle_audio_socket(self, session_id: str, websocket: WebSocket) -> None:
        session = self.sessions.get(session_id)
        if await close_unknown_session(session, websocket):
            return

        session = await reset_existing_audio_socket_session(session_id, session, self.sessions, self._cleanup_session, websocket)
        if session is None:
            return

        await websocket.accept()
        session.audio_socket = websocket
        buffered = np.asarray([], dtype=np.float32)
        await self._debug(session, "audio websocket connected")

        try:
            while True:
                message = await websocket.receive()
                session.received_messages += 1
                raw = message.get("bytes")
                if session.received_messages <= 4:
                    size = len(raw) if raw else 0
                    await self._debug(session, f"received blob #{session.received_messages} bytes={size}")
                async def infer_for_chunk(*, chunk: np.ndarray, audio_preprocessor: object | None) -> object:
                    return await run_chunk_inference(
                        chunk=chunk,
                        pipe=self.ctx.pipe,
                        generate_kwargs=self.ctx.generate_kwargs,
                        audio_preprocessor=audio_preprocessor,
                        infer_lock=self.ctx.infer_lock,
                        transcribe_fn=transcribe_chunk,
                        should_suppress_fn=should_suppress_transcript,
                        likely_reason_details_fn=likely_reason_details,
                        to_thread_fn=asyncio.to_thread,
                    )

                async def process_chunks(*, session: SessionState, chunks: list[np.ndarray]) -> None:
                    await process_prepared_chunks(
                        session=session,
                        chunks=chunks,
                        target_sample_rate=TARGET_SAMPLE_RATE,
                        append_capture_audio_fn=append_capture_audio,
                        run_chunk_inference_fn=infer_for_chunk,
                        debug_fn=self._debug,
                    )

                result = await process_audio_message(
                    session=session,
                    message=message,
                    buffered_audio=buffered,
                    verbose=self.ctx.verbose,
                    start_time=self.ctx.start_time,
                    logger=log,
                    debug_fn=self._debug,
                    decode_audio_message_fn=self._decode_audio_message,
                    prepare_stream_chunks_fn=prepare_stream_chunks,
                    process_prepared_chunks_fn=process_chunks,
                    cleanup_session_fn=self._cleanup_session,
                )
                buffered = result.buffered_audio
                if result.stop:
                    return
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await session.queue.put(f"[server error] Audio stream failed: {exc}")
            except Exception:
                pass
        finally:
            await self._cleanup_session(session)

    def _decode_audio_message(self, session: Session, message: dict[str, object]) -> tuple[np.ndarray, int] | None:
        return decode_session_message(
            session=session,
            message=message,
            max_encoded_buffer_bytes=MAX_ENCODED_BUFFER_BYTES,
            decode_message=lambda **kwargs: decode_audio_message(
                decode_payload=self._try_decode_bytes,
                **kwargs,
            ),
            invalid_data_error_type=av.error.InvalidDataError,
        )

    def _try_decode_bytes(self, payload: bytes, mime_type: str | None) -> tuple[np.ndarray, int] | None:
        return try_decode_bytes(payload=payload, mime_type=mime_type)

    async def _event_stream(self, session: SessionState) -> object:
        async for item in event_stream(queue=session.queue, ping_timeout=15.0):
            yield item

    async def _cleanup_session(self, session: SessionState) -> None:
        await cleanup_session(session=session, sessions=self.sessions, target_sample_rate=TARGET_SAMPLE_RATE)


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
        load_index_html_fn=load_index_html,
        service_factory=lambda *, ctx, index_html: AudioStreamService(
            ctx=ctx,
            index_html=index_html,
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
