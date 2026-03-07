#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai_voice import (
    TARGET_SAMPLE_RATE,
    configure_openvino_runtime_env,
    create_audio_preprocessor,
    log,
    preprocess_audio,
    resample_audio_linear,
    transcribe_chunk,
)
from voice_runtime import DeviceListRequested, PipelineSetupError, create_whisper_runtime, likely_reason_details

DEFAULT_CHUNK_SECONDS = 1.5
DEFAULT_OVERLAP_SECONDS = 0.25
UI_PATH = pathlib.Path(__file__).resolve().parent / "web" / "index.html"
WORKLET_PATH = pathlib.Path(__file__).resolve().parent / "web" / "audio-worklet.js"


class SessionConfig(BaseModel):
    session_id: str
    save_sample: bool = False
    silence_detect: bool = True
    vad_mode: int = 2
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS
    sample_rate: int = TARGET_SAMPLE_RATE


@dataclass
class ServerContext:
    pipe: object
    generate_kwargs: dict[str, object]
    silence_detect_default: bool
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
    audio_preprocessor: object | None
    chunk_seconds: float
    overlap_seconds: float
    sample_rate: int
    capture_path: pathlib.Path | None = None
    capture_writer: wave.Wave_write | None = None
    capture_samples: int = 0
    audio_socket: WebSocket | None = None


def load_index_html(silence_detect_default: bool) -> str:
    if UI_PATH.exists():
        checked_attr = "checked" if silence_detect_default else ""
        return (
            UI_PATH.read_text(encoding="utf-8")
            .replace("__SILENCE_DETECT_DEFAULT__", checked_attr)
            .replace("__VAD_MODE_DEFAULT__", "2")
        )
    return "<!doctype html><html><body><h3>UI file missing</h3></body></html>"


def load_worklet_js() -> str:
    if WORKLET_PATH.exists():
        return WORKLET_PATH.read_text(encoding="utf-8")
    return "class MissingProcessor extends AudioWorkletProcessor { process() { return true; } } registerProcessor('pcm-resample-processor', MissingProcessor);"


def validate_chunk_config(chunk_seconds: float, overlap_seconds: float) -> None:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")


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
        silence_detect_default=args.silence_detect,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        verbose=args.verbose,
        start_time=start_time,
        infer_lock=asyncio.Lock(),
    )


class AudioStreamService:
    def __init__(self, ctx: ServerContext, index_html: str, worklet_js: str) -> None:
        self.ctx = ctx
        self.index_html = index_html
        self.worklet_js = worklet_js
        self.sessions: dict[str, Session] = {}

    def build_app(self) -> FastAPI:
        app = FastAPI(title="Browser Mic Transcriber")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return self.index_html

        @app.get("/audio-worklet.js")
        async def worklet() -> Response:
            return Response(content=self.worklet_js, media_type="application/javascript")

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
        if payload.sample_rate <= 0:
            raise HTTPException(status_code=400, detail="sample_rate must be > 0")

        try:
            audio_preprocessor = create_audio_preprocessor(payload.silence_detect, vad_mode=payload.vad_mode)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {exc}") from exc

        self.sessions[payload.session_id] = Session(
            session_id=payload.session_id,
            queue=asyncio.Queue(maxsize=64),
            save_sample=payload.save_sample,
            silence_detect=payload.silence_detect,
            audio_preprocessor=audio_preprocessor,
            chunk_seconds=payload.chunk_seconds,
            overlap_seconds=payload.overlap_seconds,
            sample_rate=payload.sample_rate,
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
        chunk_samples = int(round(session.chunk_seconds * TARGET_SAMPLE_RATE))
        stride_samples = chunk_samples - int(round(session.overlap_seconds * TARGET_SAMPLE_RATE))
        if stride_samples <= 0:
            await session.queue.put("[server error] Invalid chunk configuration.")
            await self._cleanup_session(session)
            return

        try:
            while True:
                message = await websocket.receive()
                audio = self._decode_audio_message(message, session.sample_rate)
                if audio is None or audio.size == 0:
                    continue

                out_path = self._append_capture_audio(session, audio)
                if out_path is not None and session.capture_samples == int(audio.size):
                    await session.queue.put(f"[server] recording WAV capture: {out_path}")
                buffered = np.concatenate((buffered, audio))

                while buffered.shape[0] >= chunk_samples:
                    chunk = buffered[:chunk_samples]
                    buffered = buffered[stride_samples:]
                    if np.max(np.abs(chunk), initial=0.0) < 1e-4:
                        continue
                    try:
                        chunk = preprocess_audio(chunk, session.audio_preprocessor, self.ctx.verbose, self.ctx.start_time)
                    except Exception as exc:
                        await session.queue.put(f"[server error] Audio preprocessing failed: {exc}")
                        continue
                    if session.silence_detect and chunk.size == 0:
                        continue

                    async with self.ctx.infer_lock:
                        try:
                            text = await asyncio.to_thread(transcribe_chunk, self.ctx.pipe, chunk, self.ctx.generate_kwargs)
                        except Exception as exc:
                            details = likely_reason_details(exc)
                            await session.queue.put(f"[server error] Live transcription failed: {details[0]}")
                            continue
                    if text:
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

    def _decode_audio_message(self, message: dict[str, object], input_sample_rate: int) -> np.ndarray | None:
        raw = message.get("bytes")
        if not raw:
            return None
        audio = np.frombuffer(raw, dtype=np.float32).astype(np.float32, copy=False)
        if input_sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_audio_linear(audio, input_sample_rate, TARGET_SAMPLE_RATE)
        return audio

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
            duration = session.capture_samples / float(TARGET_SAMPLE_RATE)
            try:
                session.queue.put_nowait(f"[server] saved WAV capture: {saved_path} ({duration:.2f}s)")
            except Exception:
                pass

        self.sessions.pop(session.session_id, None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser page and transcribe client microphone over AudioWorklet/WebSocket.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server (default: 8000).")
    parser.add_argument("--tls-certfile", type=pathlib.Path, default=None, help="TLS certificate file (PEM).")
    parser.add_argument("--tls-keyfile", type=pathlib.Path, default=None, help="TLS private key file (PEM).")
    parser.add_argument("--device", default="NPU,GPU,CPU", help="Device preference order using NPU,GPU,CPU, or 'list' to print detected devices.")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help="Optional model directory. Defaults: NPU->whisper-base.en-int8-ov, GPU/CPU->whisper-tiny-fp16-ov.",
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
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=2, help="Default WebRTC VAD aggressiveness mode for browser sessions.")
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS, help=f"Chunk duration in seconds for server-side transcription windows (default: {DEFAULT_CHUNK_SECONDS}).")
    parser.add_argument("--overlap-seconds", type=float, default=DEFAULT_OVERLAP_SECONDS, help=f"Chunk overlap in seconds to preserve context across windows (default: {DEFAULT_OVERLAP_SECONDS}).")
    parser.add_argument("--profile", action="store_true", help="Enable py-spy profiling for this run.")
    parser.add_argument("--profile-output", type=pathlib.Path, default=None, help="Optional py-spy output SVG path (default: profiles/<timestamp>.svg).")
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")
    return parser.parse_args(argv)


def validate_tls_args(args: argparse.Namespace) -> None:
    if (args.tls_certfile is None) != (args.tls_keyfile is None):
        raise ValueError("--tls-certfile and --tls-keyfile must be provided together.")
    if args.tls_certfile is not None and not args.tls_certfile.exists():
        raise ValueError(f"TLS certificate file not found: {args.tls_certfile}")
    if args.tls_keyfile is not None and not args.tls_keyfile.exists():
        raise ValueError(f"TLS key file not found: {args.tls_keyfile}")


def main(argv: list[str] | None = None) -> int:
    enable_loopback_only_network()
    args = parse_args(argv)
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice-web",
        output_path=args.profile_output,
    )
    try:
        try:
            validate_tls_args(args)
            validate_chunk_config(args.chunk_seconds, args.overlap_seconds)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        start_time = time.perf_counter()
        try:
            ctx = create_context(args, start_time)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3

        try:
            import uvicorn
        except Exception as exc:
            print(f"Error: uvicorn is not available: {exc}", file=sys.stderr)
            return 3

        service = AudioStreamService(ctx=ctx, index_html=load_index_html(ctx.silence_detect_default), worklet_js=load_worklet_js())
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


if __name__ == "__main__":
    raise SystemExit(main())
