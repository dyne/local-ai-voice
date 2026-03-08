from __future__ import annotations

import asyncio
import pathlib

import pytest
from fastapi import HTTPException

from browser_webrtc import AudioStreamService, ServerContext, SessionConfig
from local_ai.slices.voice.web_ui.session_state import SessionState


class FakeSocket:
    def __init__(self) -> None:
        self.closed: list[tuple[int | None, str | None]] = []

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed.append((code, reason))


def make_context() -> ServerContext:
    return ServerContext(
        pipe=object(),
        generate_kwargs={},
        selected_device="CPU",
        model_dir=pathlib.Path("model"),
        silence_detect_default=True,
        vad_mode_default=3,
        chunk_seconds=1.5,
        overlap_seconds=0.0,
        verbose=False,
        start_time=0.0,
        infer_lock=asyncio.Lock(),
    )


def make_session(session_id: str = "abc") -> SessionState:
    return SessionState(
        session_id=session_id,
        queue=asyncio.Queue(maxsize=10),
        save_sample=False,
        silence_detect=True,
        debug=False,
        audio_preprocessor=None,
        chunk_seconds=1.0,
        overlap_seconds=0.0,
    )


@pytest.mark.anyio
async def test_create_session_replaces_existing_session(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AudioStreamService(make_context(), "<html></html>")
    old_session = make_session("abc")
    service.sessions["abc"] = old_session
    cleaned: list[str] = []

    async def fake_cleanup(session: SessionState) -> None:
        cleaned.append(session.session_id)
        service.sessions.pop(session.session_id, None)

    monkeypatch.setattr(service, "_cleanup_session", fake_cleanup)

    response = await service._create_session(
        SessionConfig(
            session_id="abc",
            save_sample=False,
            silence_detect=True,
            debug=False,
            vad_mode=3,
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            mime_type="audio/webm",
            audio_bitrate=48000,
        )
    )

    assert cleaned == ["abc"]
    assert response.body == b'{"ok":true}'
    assert service.sessions["abc"].mime_type == "audio/webm"


@pytest.mark.anyio
async def test_create_session_translates_validation_errors() -> None:
    service = AudioStreamService(make_context(), "<html></html>")

    with pytest.raises(HTTPException) as exc:
        await service._create_session(
            SessionConfig(
                session_id="abc",
                save_sample=False,
                silence_detect=True,
                debug=False,
                vad_mode=3,
                chunk_seconds=0.0,
                overlap_seconds=0.0,
                mime_type=None,
                audio_bitrate=48000,
            )
        )

    assert exc.value.status_code == 400


@pytest.mark.anyio
async def test_cleanup_session_closes_socket_emits_saved_capture_and_removes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AudioStreamService(make_context(), "<html></html>")
    session = make_session("abc")
    session.audio_socket = FakeSocket()
    session.capture_path = pathlib.Path("capture.wav")
    session.capture_sample_rate = 16000
    session.capture_samples = 32000
    service.sessions["abc"] = session

    async def fake_cleanup(session: SessionState, sessions: dict[str, SessionState], target_sample_rate: int) -> None:
        if session.audio_socket is not None:
            await session.audio_socket.close()
            session.audio_socket = None
        session.queue.put_nowait("[server] saved WAV capture: capture.wav (2.00s)")
        sessions.pop(session.session_id, None)

    monkeypatch.setattr("browser_webrtc.cleanup_session", fake_cleanup)

    await service._cleanup_session(session)

    assert session.audio_socket is None
    assert service.sessions == {}
    assert session.queue.get_nowait() == "[server] saved WAV capture: capture.wav (2.00s)"


@pytest.mark.anyio
async def test_decode_audio_message_resets_session_state_on_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AudioStreamService(make_context(), "<html></html>")
    session = make_session("abc")
    session.encoded_buffer = bytearray(b"abc")
    session.decoded_sample_cursor = 5
    session.decoded_sample_rate = 16000

    monkeypatch.setattr(
        "browser_webrtc.decode_audio_message",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Encoded audio buffer overflow; browser stream is not decodable.")),
    )

    with pytest.raises(RuntimeError, match="Encoded audio buffer overflow"):
        service._decode_audio_message(session, {"bytes": b"more"})

    assert session.encoded_buffer == bytearray()
    assert session.decoded_sample_cursor == 0
    assert session.decoded_sample_rate is None


@pytest.mark.anyio
async def test_handle_audio_socket_closes_unknown_session() -> None:
    service = AudioStreamService(make_context(), "<html></html>")
    websocket = FakeSocket()

    await service._handle_audio_socket("missing", websocket)

    assert websocket.closed == [(4404, "Unknown session")]


@pytest.mark.anyio
async def test_handle_audio_socket_closes_when_existing_socket_session_is_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AudioStreamService(make_context(), "<html></html>")
    session = make_session("abc")
    session.audio_socket = FakeSocket()
    service.sessions["abc"] = session
    websocket = FakeSocket()

    async def fake_cleanup(current: SessionState) -> None:
        service.sessions.pop(current.session_id, None)

    monkeypatch.setattr(service, "_cleanup_session", fake_cleanup)

    await service._handle_audio_socket("abc", websocket)

    assert websocket.closed == [(4409, "Session reset")]
