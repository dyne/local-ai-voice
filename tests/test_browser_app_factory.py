from __future__ import annotations

import pathlib

from fastapi.testclient import TestClient

from local_ai.slices.voice.web_ui.app_factory import build_browser_app


def test_build_browser_app_registers_expected_routes() -> None:
    app = build_browser_app(
        index_html="<html></html>",
        create_session_handler=lambda payload: None,
        audio_handler=lambda session_id, websocket: None,
        events_handler=lambda session_id: None,
        close_session_handler=lambda session_id: None,
    )

    routes = {(tuple(sorted(getattr(route, "methods", []) or [])), route.path) for route in app.routes}

    assert (("GET",), "/") in routes
    assert (("POST",), "/session") in routes
    assert (("GET",), "/events/{session_id}") in routes
    assert (("DELETE",), "/session/{session_id}") in routes
    assert ((), "/audio/{session_id}") in routes


def test_build_browser_app_serves_static_assets_when_directory_exists(tmp_path: pathlib.Path) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('ready');", encoding="utf-8")

    app = build_browser_app(
        index_html="<html><body>ok</body></html>",
        static_assets_dir=assets_dir,
        create_session_handler=lambda payload: None,
        audio_handler=lambda session_id, websocket: None,
        events_handler=lambda session_id: None,
        close_session_handler=lambda session_id: None,
    )

    client = TestClient(app)
    response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "console.log('ready');" in response.text
