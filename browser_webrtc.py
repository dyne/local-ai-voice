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
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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


class Offer(BaseModel):
    sdp: str
    type: str
    session_id: str
    save_sample: bool = False
    silence_detect: bool = True
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS


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
    pc: RTCPeerConnection
    queue: asyncio.Queue[str]
    save_sample: bool
    silence_detect: bool
    audio_preprocessor: object | None
    chunk_seconds: float
    overlap_seconds: float
    capture_path: pathlib.Path | None = None
    capture_writer: wave.Wave_write | None = None
    capture_samples: int = 0
    worker: asyncio.Task | None = None


def load_index_html(silence_detect_default: bool) -> str:
    if UI_PATH.exists():
        checked_attr = "checked" if silence_detect_default else ""
        return UI_PATH.read_text(encoding="utf-8").replace("__SILENCE_DETECT_DEFAULT__", checked_attr)
    return "<!doctype html><html><body><h3>UI file missing</h3></body></html>"


def validate_chunk_config(chunk_seconds: float, overlap_seconds: float) -> None:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")


def audio_frame_to_float32_mono(frame: object) -> np.ndarray:
    arr = frame.to_ndarray()
    fmt_name = str(getattr(getattr(frame, "format", None), "name", "")).lower()

    if np.issubdtype(arr.dtype, np.unsignedinteger):
        info = np.iinfo(arr.dtype)
        center = float((info.max + 1) // 2)
        data = (arr.astype(np.float32) - center) / center
    elif np.issubdtype(arr.dtype, np.signedinteger):
        if fmt_name in {"s16", "s16p"}:
            denom = 32768.0
        elif fmt_name in {"s32", "s32p"}:
            peak = float(np.max(np.abs(arr), initial=0))
            denom = 32768.0 if peak <= 65536.0 else 2147483648.0
        else:
            info = np.iinfo(arr.dtype)
            denom = float(max(abs(info.min), info.max))
        data = arr.astype(np.float32) / denom
    else:
        data = arr.astype(np.float32)

    if data.ndim == 1:
        mono = data
    elif data.ndim == 2:
        mono = data.mean(axis=0) if data.shape[0] <= data.shape[1] else data.mean(axis=1)
    else:
        mono = data.reshape(-1)
    return np.asarray(mono, dtype=np.float32)


def frame_to_target_audio(frame: object, resampler: object | None) -> list[np.ndarray]:
    if resampler is None:
        sample_rate = int(getattr(frame, "sample_rate", TARGET_SAMPLE_RATE))
        audio = audio_frame_to_float32_mono(frame)
        if sample_rate > 0 and sample_rate != TARGET_SAMPLE_RATE:
            audio = resample_audio_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        return [audio] if audio.size else []

    out_frames = resampler.resample(frame)
    if out_frames is None:
        return []
    if not isinstance(out_frames, list):
        out_frames = [out_frames]

    chunks: list[np.ndarray] = []
    for out in out_frames:
        arr = out.to_ndarray()
        if arr.ndim == 1:
            audio = np.asarray(arr, dtype=np.float32)
        elif arr.ndim == 2:
            audio = np.asarray(arr[0], dtype=np.float32) if arr.shape[0] == 1 else np.asarray(arr.mean(axis=0), dtype=np.float32)
        else:
            audio = np.asarray(arr.reshape(-1), dtype=np.float32)
        if audio.size:
            chunks.append(audio)
    return chunks


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


class WebRTCService:
    def __init__(self, ctx: ServerContext, index_html: str) -> None:
        self.ctx = ctx
        self.index_html = index_html
        self.sessions: dict[str, Session] = {}

    def build_app(self) -> FastAPI:
        app = FastAPI(title="Browser Mic Transcriber")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return self.index_html

        @app.post("/offer")
        async def offer(payload: Offer) -> JSONResponse:
            return await self._handle_offer(payload)

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

    async def _handle_offer(self, payload: Offer) -> JSONResponse:
        if payload.session_id in self.sessions:
            await self._cleanup_session(self.sessions[payload.session_id])

        try:
            validate_chunk_config(payload.chunk_seconds, payload.overlap_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session = Session(
            session_id=payload.session_id,
            pc=RTCPeerConnection(),
            queue=asyncio.Queue(maxsize=64),
            save_sample=payload.save_sample,
            silence_detect=payload.silence_detect,
            audio_preprocessor=None,
            chunk_seconds=payload.chunk_seconds,
            overlap_seconds=payload.overlap_seconds,
        )
        try:
            session.audio_preprocessor = create_audio_preprocessor(payload.silence_detect)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {exc}") from exc
        self.sessions[payload.session_id] = session
        self._register_peer_handlers(session)

        try:
            await session.pc.setRemoteDescription(RTCSessionDescription(sdp=payload.sdp, type=payload.type))
            answer = await session.pc.createAnswer()
            await session.pc.setLocalDescription(answer)
            return JSONResponse({"sdp": session.pc.localDescription.sdp, "type": session.pc.localDescription.type})
        except Exception as exc:
            await self._cleanup_session(session)
            raise HTTPException(status_code=400, detail=f"WebRTC negotiation failed: {exc}") from exc

    def _register_peer_handlers(self, session: Session) -> None:
        @session.pc.on("track")
        async def on_track(track: object) -> None:
            if getattr(track, "kind", None) != "audio":
                return
            if session.worker is not None:
                session.worker.cancel()
            session.worker = asyncio.create_task(self._run_audio_worker(track, session))

        @session.pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if session.pc.connectionState in {"failed", "closed", "disconnected"}:
                await self._cleanup_session(session)

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
        if session.worker is not None:
            session.worker.cancel()
            try:
                await session.worker
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        saved_path = self._close_capture_writer(session)
        if saved_path is not None:
            duration = session.capture_samples / float(TARGET_SAMPLE_RATE)
            try:
                session.queue.put_nowait(f"[server] saved WAV capture: {saved_path} ({duration:.2f}s)")
            except Exception:
                pass

        await session.pc.close()
        self.sessions.pop(session.session_id, None)

    def _create_resampler(self) -> tuple[object | None, str | None]:
        try:
            from av.audio.resampler import AudioResampler

            return AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE), None
        except Exception as exc:
            return None, f"[server] audio resampler unavailable, using fallback conversion: {exc}"

    async def _run_audio_worker(self, track: object, session: Session) -> None:
        chunk_seconds = session.chunk_seconds if session.chunk_seconds > 0 else self.ctx.chunk_seconds
        overlap_seconds = session.overlap_seconds if session.overlap_seconds >= 0 else self.ctx.overlap_seconds
        chunk_samples = int(round(chunk_seconds * TARGET_SAMPLE_RATE))
        stride_samples = chunk_samples - int(round(overlap_seconds * TARGET_SAMPLE_RATE))
        if stride_samples <= 0:
            await session.queue.put("[server error] Invalid chunk configuration.")
            return

        resampler, warning = self._create_resampler()
        if warning is not None:
            await session.queue.put(warning)

        buffered = np.asarray([], dtype=np.float32)
        while True:
            frame = await track.recv()
            chunks = frame_to_target_audio(frame, resampler)
            if not chunks:
                continue

            for audio in chunks:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser page and transcribe client microphone over WebRTC.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the HTTP server (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server (default: 8000).")
    parser.add_argument("--tls-certfile", type=pathlib.Path, default=None, help="TLS certificate file (PEM).")
    parser.add_argument("--tls-keyfile", type=pathlib.Path, default=None, help="TLS private key file (PEM).")
    parser.add_argument(
        "--device",
        default="NPU,GPU,CPU",
        help="Device preference order using NPU,GPU,CPU, or 'list' to print detected devices.",
    )
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
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk duration in seconds for server-side transcription windows (default: {DEFAULT_CHUNK_SECONDS}).",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=DEFAULT_OVERLAP_SECONDS,
        help=f"Chunk overlap in seconds to preserve context across windows (default: {DEFAULT_OVERLAP_SECONDS}).",
    )
    parser.add_argument("--profile", action="store_true", help="Enable py-spy profiling for this run.")
    parser.add_argument(
        "--profile-output",
        type=pathlib.Path,
        default=None,
        help="Optional py-spy output SVG path (default: profiles/<timestamp>.svg).",
    )
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
        label="local-ai-voice-webrtc",
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

        service = WebRTCService(ctx=ctx, index_html=load_index_html(ctx.silence_detect_default))
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
