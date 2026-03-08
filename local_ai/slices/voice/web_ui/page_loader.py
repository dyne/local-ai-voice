from __future__ import annotations

import pathlib


def load_index_html(*, ui_path: pathlib.Path, silence_detect_default: bool, vad_mode_default: int) -> str:
    if ui_path.exists():
        checked_attr = "checked" if silence_detect_default else ""
        return (
            ui_path.read_text(encoding="utf-8")
            .replace("__SILENCE_DETECT_DEFAULT__", checked_attr)
            .replace("__VAD_MODE_DEFAULT__", str(vad_mode_default))
        )
    return "<!doctype html><html><body><h3>UI file missing</h3></body></html>"
