from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse


def build_browser_app(
    *,
    index_html: str,
    create_session_handler: Callable[[object], Awaitable[object]],
    audio_handler: Callable[[str, WebSocket], Awaitable[None]],
    events_handler: Callable[[str], Awaitable[object]],
    close_session_handler: Callable[[str], Awaitable[object]],
) -> FastAPI:
    app = FastAPI(title="Browser Mic Transcriber")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return index_html

    @app.post("/session")
    async def create_session(payload: object) -> object:
        return await create_session_handler(payload)

    @app.websocket("/audio/{session_id}")
    async def audio(session_id: str, websocket: WebSocket) -> None:
        await audio_handler(session_id, websocket)

    @app.get("/events/{session_id}")
    async def events(session_id: str) -> object:
        return await events_handler(session_id)

    @app.delete("/session/{session_id}")
    async def close_session(session_id: str) -> object:
        return await close_session_handler(session_id)

    return app
