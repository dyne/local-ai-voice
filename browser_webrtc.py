#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import io
import pathlib
import socket
import sys
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import av
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from local_ai.slices.voice.shared.audio_processing import (
    TARGET_SAMPLE_RATE,
    create_audio_preprocessor,
    normalize_audio_format,
)
from local_ai.slices.voice.transcribe_stream.buffer_decoder import decode_audio_message
from local_ai.slices.voice.shared.transcript_policy import should_suppress_transcript, transcribe_chunk
from local_ai.slices.voice.transcribe_stream.request import TranscribeStreamChunkRequest
from local_ai.slices.voice.transcribe_stream.service import prepare_stream_chunks
from local_ai.slices.voice.web_ui.server_config import (
    desktop_host,
    mime_type_to_av_format,
    validate_chunk_config,
    validate_tls_paths,
)
from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai_voice import (
    configure_openvino_runtime_env,
    log,
)
from local_ai.infrastructure.openvino.whisper import create_whisper_runtime, likely_reason_details
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError

DEFAULT_CHUNK_SECONDS = 1.5
DEFAULT_OVERLAP_SECONDS = 0.0
DEFAULT_AUDIO_BITRATE = 48000
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


@dataclass
class Session:
    session_id: str
    queue: asyncio.Queue[str]
    save_sample: bool
    silence_detect: bool
    debug: bool
    audio_preprocessor: object | None
    chunk_seconds: float
    overlap_seconds: float
    stream_sample_rate: int | None = None
    mime_type: str | None = None
    audio_bitrate: int = DEFAULT_AUDIO_BITRATE
    capture_path: pathlib.Path | None = None
    capture_writer: wave.Wave_write | None = None
    capture_sample_rate: int | None = None
    capture_samples: int = 0
    audio_socket: WebSocket | None = None
    encoded_buffer: bytearray = field(default_factory=bytearray)
    decoded_sample_cursor: int = 0
    decoded_sample_rate: int | None = None
    received_messages: int = 0
    decode_attempts: int = 0
    decoded_messages: int = 0
    model_chunks: int = 0
    debug_messages_sent: int = 0


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
    configure_openvino_runtime_env()
    try:
        runtime = create_whisper_runtime(
            args=args,
            base_dir=pathlib.Path(__file__).resolve().parent,
            logger=log,
            verbose=args.verbose,
            start_time=start_time,
        )
    except DeviceListRequested as exc:
        if not exc.devices:
            raise RuntimeError("No OpenVINO devices detected.")
        raise RuntimeError("Devices: " + ", ".join(exc.devices))
    except PipelineSetupError as exc:
        raise RuntimeError(f"{exc.reason} {' '.join(exc.details)}")

    return ServerContext(
        pipe=runtime.pipe,
        generate_kwargs=runtime.generate_kwargs,
        selected_device=runtime.selected_device,
        model_dir=runtime.model_dir,
        silence_detect_default=args.silence_detect,
        vad_mode_default=args.vad_mode,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        verbose=args.verbose,
        start_time=start_time,
        infer_lock=asyncio.Lock(),
    )


class AudioStreamService:
    def __init__(self, ctx: ServerContext, index_html: str) -> None:
        self.ctx = ctx
        self.index_html = index_html
        self.sessions: dict[str, Session] = {}

    async def _debug(self, session: Session, message: str, limit: int = 12) -> None:
        if not session.debug:
            return
        if session.debug_messages_sent >= limit:
            return
        session.debug_messages_sent += 1
        await session.queue.put(f"[debug] {message}")

    def build_app(self) -> FastAPI:
        app = FastAPI(title="Browser Mic Transcriber")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return self.index_html

        @app.post("/session")
        async def create_session(payload: SessionConfig) -> JSONResponse:
            return await self._create_session(payload)

        @app.websocket("/audio/{session_id}")
        async def audio(session_id: str, websocket: WebSocket) -> None:
            await self._handle_audio_socket(session_id, websocket)

        @app.get("/events/{session_id}")
        async def events(session_id: str) -> StreamingResponse:
            session = self.sessions.get(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Unknown session")
            return StreamingResponse(self._event_stream(session), media_type="text/event-stream")

        @app.delete("/session/{session_id}")
        async def close_session(session_id: str) -> JSONResponse:
            session = self.sessions.get(session_id)
            if session is not None:
                await self._cleanup_session(session)
            return JSONResponse({"ok": True})

        return app

    async def _create_session(self, payload: SessionConfig) -> JSONResponse:
        if payload.session_id in self.sessions:
            await self._cleanup_session(self.sessions[payload.session_id])

        try:
            validate_chunk_config(payload.chunk_seconds, payload.overlap_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if payload.audio_bitrate <= 0:
            raise HTTPException(status_code=400, detail="audio_bitrate must be > 0")

        try:
            audio_preprocessor = create_audio_preprocessor(payload.silence_detect, vad_mode=payload.vad_mode)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {exc}") from exc

        self.sessions[payload.session_id] = Session(
            session_id=payload.session_id,
            queue=asyncio.Queue(maxsize=64),
            save_sample=payload.save_sample,
            silence_detect=payload.silence_detect,
            debug=payload.debug,
            audio_preprocessor=audio_preprocessor,
            chunk_seconds=payload.chunk_seconds,
            overlap_seconds=payload.overlap_seconds,
            mime_type=payload.mime_type,
            audio_bitrate=payload.audio_bitrate,
        )
        await self._debug(
            self.sessions[payload.session_id],
            f"session created mime={payload.mime_type or 'unknown'} bitrate={payload.audio_bitrate} chunk={payload.chunk_seconds:.2f}s overlap={payload.overlap_seconds:.2f}s save_sample={payload.save_sample}",
        )
        return JSONResponse({"ok": True})

    async def _handle_audio_socket(self, session_id: str, websocket: WebSocket) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="Unknown session")
            return

        if session.audio_socket is not None:
            await self._cleanup_session(session)
            session = self.sessions.get(session_id)
            if session is None:
                await websocket.close(code=4409, reason="Session reset")
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
                decoded = self._decode_audio_message(session, message)
                if decoded is None:
                    if session.received_messages <= 4:
                        await self._debug(session, f"blob #{session.received_messages} not decodable yet buffer={len(session.encoded_buffer)}")
                    continue
                audio, sample_rate = decoded
                session.decoded_messages += 1
                if session.decoded_messages <= 4:
                    await self._debug(
                        session,
                        f"decoded chunk #{session.decoded_messages} samples={audio.size} sample_rate={sample_rate} buffer_after={len(session.encoded_buffer)}",
                    )
                if audio.size == 0:
                    continue
                prepared = prepare_stream_chunks(
                    request=TranscribeStreamChunkRequest(
                        incoming_audio=audio,
                        incoming_sample_rate=sample_rate,
                        buffered_audio=buffered,
                        current_stream_sample_rate=session.stream_sample_rate,
                        chunk_seconds=session.chunk_seconds,
                        overlap_seconds=session.overlap_seconds,
                        silence_detect=session.silence_detect,
                        audio_preprocessor=session.audio_preprocessor,
                        verbose=self.ctx.verbose,
                        start=self.ctx.start_time,
                    ),
                    logger=log,
                )
                buffered = prepared.buffered_audio
                session.stream_sample_rate = prepared.stream_sample_rate
                if prepared.error is not None:
                    await session.queue.put("[server error] Invalid chunk configuration.")
                    await self._cleanup_session(session)
                    return

                if prepared.rejected_by_preprocessor and session.model_chunks == 0:
                    await self._debug(session, "chunk rejected by preprocessing/VAD before model input")

                for chunk in prepared.model_inputs:
                    out_path = self._append_capture_audio(session, chunk)
                    if out_path is not None and session.capture_samples == int(chunk.size):
                        await session.queue.put(f"[server] recording WAV capture: {out_path}")
                    session.model_chunks += 1
                    if session.model_chunks <= 4:
                        await self._debug(session, f"model chunk #{session.model_chunks} samples={chunk.size} sample_rate={TARGET_SAMPLE_RATE}")

                    async with self.ctx.infer_lock:
                        try:
                            text = await asyncio.to_thread(transcribe_chunk, self.ctx.pipe, chunk, self.ctx.generate_kwargs)
                        except Exception as exc:
                            details = likely_reason_details(exc)
                            await session.queue.put(f"[server error] Live transcription failed: {details[0]}")
                            continue
                    if text and not should_suppress_transcript(text, session.audio_preprocessor):
                        await session.queue.put(text)
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
        raw = message.get("bytes")
        if raw:
            session.decode_attempts += 1
        try:
            decoded = decode_audio_message(
                raw=raw if isinstance(raw, (bytes, bytearray)) else None,
                encoded_buffer=session.encoded_buffer,
                decoded_sample_cursor=session.decoded_sample_cursor,
                decoded_sample_rate=session.decoded_sample_rate,
                mime_type=session.mime_type,
                max_encoded_buffer_bytes=MAX_ENCODED_BUFFER_BYTES,
                decode_payload=self._try_decode_bytes,
                invalid_data_error_type=av.error.InvalidDataError,
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

    def _try_decode_bytes(self, payload: bytes, mime_type: str | None) -> tuple[np.ndarray, int] | None:
        format_hint = mime_type_to_av_format(mime_type)
        with av.open(io.BytesIO(payload), format=format_hint) as container:
            decoded_frames: list[np.ndarray] = []
            sample_rate: int | None = None
            for frame in container.decode(audio=0):
                mono = self._decode_audio_frame(frame)
                if mono.size == 0:
                    continue
                decoded_frames.append(mono)
                sample_rate = int(frame.sample_rate)
        if not decoded_frames or sample_rate is None:
            return None
        return np.concatenate(decoded_frames), sample_rate

    def _decode_audio_frame(self, frame: av.AudioFrame) -> np.ndarray:
        array = frame.to_ndarray()
        if array.ndim == 2:
            audio = array.mean(axis=0)
        else:
            audio = array
        audio = normalize_audio_format(audio)
        if np.issubdtype(array.dtype, np.integer):
            max_value = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
            audio = audio / float(max_value)
        return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)

    async def _event_stream(self, session: Session) -> object:
        while True:
            try:
                line = await asyncio.wait_for(session.queue.get(), timeout=15.0)
                yield f"data: {line}\n\n"
            except asyncio.TimeoutError:
                yield "event: ping\ndata: keepalive\n\n"
            except asyncio.CancelledError:
                break

    def _ensure_capture_writer(self, session: Session) -> pathlib.Path:
        if session.capture_writer is None:
            captures_dir = pathlib.Path.cwd() / "captures"
            captures_dir.mkdir(parents=True, exist_ok=True)
            session.capture_path = captures_dir / f"capture_{session.session_id}_{int(time.time())}.wav"
            writer = wave.open(str(session.capture_path), "wb")
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(TARGET_SAMPLE_RATE)
            session.capture_writer = writer
            session.capture_sample_rate = TARGET_SAMPLE_RATE
        return session.capture_path

    def _append_capture_audio(self, session: Session, audio: np.ndarray) -> pathlib.Path | None:
        if not session.save_sample or audio.size == 0:
            return None
        out_path = self._ensure_capture_writer(session)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        session.capture_writer.writeframes(pcm16.tobytes())
        session.capture_samples += int(audio.size)
        return out_path

    def _close_capture_writer(self, session: Session) -> pathlib.Path | None:
        if session.capture_writer is None:
            return None
        try:
            session.capture_writer.close()
        finally:
            session.capture_writer = None
        return session.capture_path

    async def _cleanup_session(self, session: Session) -> None:
        if session.audio_socket is not None:
            try:
                await session.audio_socket.close()
            except Exception:
                pass
            session.audio_socket = None

        saved_path = self._close_capture_writer(session)
        if saved_path is not None:
            capture_rate = float(session.capture_sample_rate or TARGET_SAMPLE_RATE)
            duration = session.capture_samples / capture_rate
            try:
                session.queue.put_nowait(f"[server] saved WAV capture: {saved_path} ({duration:.2f}s)")
            except Exception:
                pass

        self.sessions.pop(session.session_id, None)


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
    validate_tls_paths(args.tls_certfile, args.tls_keyfile)
    validate_chunk_config(args.chunk_seconds, args.overlap_seconds)
    start_time = time.perf_counter()
    ctx = create_context(args, start_time)
    service = AudioStreamService(
        ctx=ctx,
        index_html=load_index_html(ctx.silence_detect_default, ctx.vad_mode_default),
    )
    return ctx, service, start_time


def run_server(args: argparse.Namespace) -> int:
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice-web",
        output_path=args.profile_output,
    )
    try:
        try:
            ctx, service, start_time = prepare_server(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3

        print(f"Using device: {ctx.selected_device}", file=sys.stderr, flush=True)
        print(f"Using model: {ctx.model_dir}", file=sys.stderr, flush=True)
        enable_loopback_only_network()

        try:
            import uvicorn
        except Exception as exc:
            print(f"Error: uvicorn is not available: {exc}", file=sys.stderr)
            return 3

        scheme = "https" if args.tls_certfile is not None else "http"
        log(f"Starting server on {scheme}://{args.host}:{args.port}", args.verbose, start_time)
        uvicorn.run(
            service.build_app(),
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_certfile=str(args.tls_certfile) if args.tls_certfile is not None else None,
            ssl_keyfile=str(args.tls_keyfile) if args.tls_keyfile is not None else None,
        )
        return 0
    finally:
        stop_py_spy_profile(profile_session)


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def wait_for_server(host: str, port: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local server on {host}:{port}")


def _print_fallback_url(args: argparse.Namespace) -> None:
    scheme = "https" if args.tls_certfile is not None else "http"
    print(f"Desktop UI unavailable. Open {scheme}://{args.host}:{args.port} in a browser.", file=sys.stderr, flush=True)


def run_desktop(args: argparse.Namespace) -> int:
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice-webview",
        output_path=args.profile_output,
    )
    try:
        try:
            args.host = desktop_host()
            if args.port == 8000:
                args.port = find_free_port(args.host)
            ctx, service, start_time = prepare_server(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3

        print(f"Using device: {ctx.selected_device}", file=sys.stderr, flush=True)
        print(f"Using model: {ctx.model_dir}", file=sys.stderr, flush=True)

        try:
            import uvicorn
            import webview
        except Exception as exc:
            print(f"Desktop UI unavailable: {exc}", file=sys.stderr, flush=True)
            _print_fallback_url(args)
            return run_server(args)

        config = uvicorn.Config(
            service.build_app(),
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_certfile=str(args.tls_certfile) if args.tls_certfile is not None else None,
            ssl_keyfile=str(args.tls_keyfile) if args.tls_keyfile is not None else None,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None

        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        wait_for_server(args.host, args.port)
        enable_loopback_only_network()

        scheme = "https" if args.tls_certfile is not None else "http"
        url = f"{scheme}://{args.host}:{args.port}"
        log(f"Starting desktop UI on {url}", args.verbose, start_time)
        try:
            webview.create_window("Local AI Voice", url, width=1280, height=900)
            webview.start()
        except Exception as exc:
            print(f"Desktop UI unavailable: {exc}", file=sys.stderr, flush=True)
            _print_fallback_url(args)
            server.should_exit = True
            server_thread.join(timeout=10.0)
            return run_server(args)

        server.should_exit = True
        server_thread.join(timeout=10.0)
        if server.force_exit:
            return 3
        return 0
    finally:
        stop_py_spy_profile(profile_session)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
