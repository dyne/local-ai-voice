from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from local_ai.slices.voice.shared.audio_processing import TARGET_SAMPLE_RATE, create_audio_preprocessor
from local_ai.slices.voice.shared.transcript_policy import should_suppress_transcript, transcribe_chunk
from local_ai.slices.voice.transcribe_stream.buffer_decoder import decode_audio_message
from local_ai.slices.voice.transcribe_stream.service import prepare_stream_chunks
from local_ai.slices.voice.web_ui.app_factory import build_browser_app
from local_ai.slices.voice.web_ui.audio_decode import try_decode_bytes
from local_ai.slices.voice.web_ui.capture_store import append_capture_audio
from local_ai.slices.voice.web_ui.chunk_pipeline import process_prepared_chunks
from local_ai.slices.voice.web_ui.event_stream import event_stream
from local_ai.slices.voice.web_ui.inference_runner import run_chunk_inference
from local_ai.slices.voice.web_ui.message_processor import process_audio_message
from local_ai.slices.voice.web_ui.session_cleanup import cleanup_session
from local_ai.slices.voice.web_ui.session_decoder import decode_session_message
from local_ai.slices.voice.web_ui.session_state import DEFAULT_AUDIO_BITRATE, SessionState, create_session_state
from local_ai.slices.voice.web_ui.socket_loop import handle_audio_socket_connection

DEFAULT_CHUNK_SECONDS = 1.5
DEFAULT_OVERLAP_SECONDS = 0.0
MAX_ENCODED_BUFFER_BYTES = 4 * 1024 * 1024


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


class AudioStreamService:
    def __init__(
        self,
        ctx: ServerContext,
        index_html: str,
        static_assets_dir: pathlib.Path | None = None,
        *,
        logger: Any = None,
        likely_reason_details_fn: Any = None,
        to_thread_fn: Any = asyncio.to_thread,
    ) -> None:
        self.ctx = ctx
        self.index_html = index_html
        self.static_assets_dir = static_assets_dir
        self.logger = logger or (lambda message, verbose, start_time: None)
        self.likely_reason_details_fn = likely_reason_details_fn or (lambda exc: ())
        self.to_thread_fn = to_thread_fn
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
            static_assets_dir=self.static_assets_dir,
            create_session_handler=self._create_session,
            audio_handler=self._handle_audio_socket,
            events_handler=events,
            close_session_handler=close_session,
        )

    async def _create_session(self, payload: SessionConfig) -> JSONResponse:
        existing = self.sessions.get(payload.session_id)
        if existing is not None:
            await self._cleanup_session(existing)

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
        async def infer_for_chunk(*, chunk: np.ndarray, audio_preprocessor: object | None) -> object:
            return await run_chunk_inference(
                chunk=chunk,
                pipe=self.ctx.pipe,
                generate_kwargs=self.ctx.generate_kwargs,
                audio_preprocessor=audio_preprocessor,
                infer_lock=self.ctx.infer_lock,
                transcribe_fn=transcribe_chunk,
                should_suppress_fn=should_suppress_transcript,
                likely_reason_details_fn=self.likely_reason_details_fn,
                to_thread_fn=self.to_thread_fn,
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

        async def process_message(*, session: SessionState, message: dict[str, object], buffered_audio: np.ndarray) -> object:
            return await process_audio_message(
                session=session,
                message=message,
                buffered_audio=buffered_audio,
                verbose=self.ctx.verbose,
                start_time=self.ctx.start_time,
                logger=self.logger,
                debug_fn=self._debug,
                decode_audio_message_fn=self._decode_audio_message,
                prepare_stream_chunks_fn=prepare_stream_chunks,
                process_prepared_chunks_fn=process_chunks,
                cleanup_session_fn=self._cleanup_session,
            )

        await handle_audio_socket_connection(
            session_id=session_id,
            websocket=websocket,
            sessions=self.sessions,
            cleanup_session_fn=self._cleanup_session,
            debug_fn=self._debug,
            process_message_fn=process_message,
            websocket_disconnect_type=WebSocketDisconnect,
        )

    def _decode_audio_message(self, session: SessionState, message: dict[str, object]) -> tuple[np.ndarray, int] | None:
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
