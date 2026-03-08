from __future__ import annotations

from typing import Awaitable, Callable

from fastapi import WebSocket

from local_ai.slices.voice.web_ui.session_state import SessionState


async def close_unknown_session(session: SessionState | None, websocket: WebSocket) -> bool:
    if session is not None:
        return False
    await websocket.close(code=4404, reason="Unknown session")
    return True


async def replace_existing_session(
    session_id: str,
    sessions: dict[str, SessionState],
    cleanup_session: Callable[[SessionState], Awaitable[None]],
) -> None:
    if session_id in sessions:
        await cleanup_session(sessions[session_id])


async def reset_existing_audio_socket_session(
    session_id: str,
    session: SessionState,
    sessions: dict[str, SessionState],
    cleanup_session: Callable[[SessionState], Awaitable[None]],
    websocket: WebSocket,
) -> SessionState | None:
    if session.audio_socket is None:
        return session
    await cleanup_session(session)
    replacement = sessions.get(session_id)
    if replacement is None:
        await websocket.close(code=4409, reason="Session reset")
        return None
    return replacement
