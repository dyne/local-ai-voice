from __future__ import annotations

from local_ai.slices.voice.web_ui.capture_store import close_capture_writer
from local_ai.slices.voice.web_ui.session_state import SessionState


async def cleanup_session(
    *,
    session: SessionState,
    sessions: dict[str, SessionState],
    target_sample_rate: int,
) -> None:
    if session.audio_socket is not None:
        try:
            await session.audio_socket.close()
        except Exception:
            pass
        session.audio_socket = None

    saved_path = close_capture_writer(session)
    if saved_path is not None:
        capture_rate = float(session.capture_sample_rate or target_sample_rate)
        duration = session.capture_samples / capture_rate
        try:
            session.queue.put_nowait(f"[server] saved WAV capture: {saved_path} ({duration:.2f}s)")
        except Exception:
            pass

    sessions.pop(session.session_id, None)
