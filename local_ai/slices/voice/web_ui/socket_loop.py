from __future__ import annotations

from collections.abc import Awaitable, Callable

import numpy as np

from local_ai.slices.voice.web_ui.message_processor import MessageProcessingResult
from local_ai.slices.voice.web_ui.session_registry import close_unknown_session, reset_existing_audio_socket_session


async def handle_audio_socket_connection(
    *,
    session_id: str,
    websocket: object,
    sessions: dict[str, object],
    cleanup_session_fn: Callable[[object], Awaitable[None]],
    debug_fn: Callable[[object, str], Awaitable[None]],
    process_message_fn: Callable[..., Awaitable[MessageProcessingResult]],
    websocket_disconnect_type: type[BaseException],
) -> None:
    session = sessions.get(session_id)
    if await close_unknown_session(session, websocket):
        return

    session = await reset_existing_audio_socket_session(session_id, session, sessions, cleanup_session_fn, websocket)
    if session is None:
        return

    await websocket.accept()
    session.audio_socket = websocket
    buffered = np.asarray([], dtype=np.float32)
    await debug_fn(session, "audio websocket connected")

    try:
        while True:
            message = await websocket.receive()
            session.received_messages += 1
            raw = message.get("bytes")
            if session.received_messages <= 4:
                size = len(raw) if raw else 0
                await debug_fn(session, f"received blob #{session.received_messages} bytes={size}")
            result = await process_message_fn(
                session=session,
                message=message,
                buffered_audio=buffered,
            )
            buffered = result.buffered_audio
            if result.stop:
                return
    except websocket_disconnect_type:
        pass
    except Exception as exc:
        try:
            await session.queue.put(f"[server error] Audio stream failed: {exc}")
        except Exception:
            pass
    finally:
        await cleanup_session_fn(session)
