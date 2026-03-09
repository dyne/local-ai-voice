from __future__ import annotations

import pathlib

from browser_webrtc import resolve_ui_path


def test_resolve_ui_path_prefers_built_frontend_when_present(tmp_path: pathlib.Path) -> None:
    frontend_index = tmp_path / "frontend" / "dist" / "index.html"
    frontend_index.parent.mkdir(parents=True)
    frontend_index.write_text("<html>frontend</html>", encoding="utf-8")

    legacy_index = tmp_path / "web" / "index.html"
    legacy_index.parent.mkdir(parents=True)
    legacy_index.write_text("<html>legacy</html>", encoding="utf-8")

    assert resolve_ui_path(base_dir=tmp_path) == frontend_index


def test_resolve_ui_path_falls_back_to_legacy_web_index(tmp_path: pathlib.Path) -> None:
    legacy_index = tmp_path / "web" / "index.html"
    legacy_index.parent.mkdir(parents=True)
    legacy_index.write_text("<html>legacy</html>", encoding="utf-8")

    assert resolve_ui_path(base_dir=tmp_path) == legacy_index
