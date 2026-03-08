from __future__ import annotations

import pathlib

from local_ai.slices.voice.web_ui.page_loader import load_index_html


def test_load_index_html_replaces_defaults(tmp_path: pathlib.Path) -> None:
    ui_path = tmp_path / "index.html"
    ui_path.write_text(
        "<input __SILENCE_DETECT_DEFAULT__ /><span>__VAD_MODE_DEFAULT__</span>",
        encoding="utf-8",
    )

    html = load_index_html(ui_path=ui_path, silence_detect_default=True, vad_mode_default=2)

    assert "__SILENCE_DETECT_DEFAULT__" not in html
    assert "__VAD_MODE_DEFAULT__" not in html
    assert "checked" in html
    assert ">2<" in html


def test_load_index_html_returns_fallback_when_missing(tmp_path: pathlib.Path) -> None:
    html = load_index_html(
        ui_path=tmp_path / "missing.html",
        silence_detect_default=False,
        vad_mode_default=3,
    )

    assert "UI file missing" in html
