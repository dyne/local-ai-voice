from __future__ import annotations

import json
import pathlib


def build_runtime_config_script(*, silence_detect_default: bool, vad_mode_default: int) -> str:
    config = {
        "silenceDetectDefault": silence_detect_default,
        "vadModeDefault": vad_mode_default,
    }
    return (
        "<script>"
        f"window.__LOCAL_AI_CONFIG__ = {json.dumps(config, separators=(',', ':'))};"
        "</script>"
    )


def inject_runtime_config(html: str, *, silence_detect_default: bool, vad_mode_default: int) -> str:
    script = build_runtime_config_script(
        silence_detect_default=silence_detect_default,
        vad_mode_default=vad_mode_default,
    )
    if "</head>" in html:
        return html.replace("</head>", f"{script}</head>", 1)
    if "</body>" in html:
        return html.replace("</body>", f"{script}</body>", 1)
    return html + script


def load_index_html(*, ui_path: pathlib.Path, silence_detect_default: bool, vad_mode_default: int) -> str:
    if ui_path.exists():
        checked_attr = "checked" if silence_detect_default else ""
        html = (
            ui_path.read_text(encoding="utf-8")
            .replace("__SILENCE_DETECT_DEFAULT__", checked_attr)
            .replace("__VAD_MODE_DEFAULT__", str(vad_mode_default))
        )
        return inject_runtime_config(
            html,
            silence_detect_default=silence_detect_default,
            vad_mode_default=vad_mode_default,
        )
    return inject_runtime_config(
        "<!doctype html><html><body><h3>UI file missing</h3></body></html>",
        silence_detect_default=silence_detect_default,
        vad_mode_default=vad_mode_default,
    )


def resolve_static_assets_dir(*, ui_path: pathlib.Path) -> pathlib.Path | None:
    assets_dir = ui_path.parent / "assets"
    if assets_dir.is_dir():
        return assets_dir
    return None
