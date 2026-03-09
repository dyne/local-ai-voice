from __future__ import annotations

from collections.abc import Awaitable, Callable
import pathlib

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def build_browser_app(
    *,
    index_html: str,
    static_assets_dir: pathlib.Path | None = None,
    create_session_handler: Callable[[object], Awaitable[object]],
    audio_handler: Callable[[str, WebSocket], Awaitable[None]],
    events_handler: Callable[[str], Awaitable[object]],
    close_session_handler: Callable[[str], Awaitable[object]],
) -> FastAPI:
    app = FastAPI(title="Browser Mic Transcriber")

    if static_assets_dir is not None and static_assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(static_assets_dir)), name="assets")

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
