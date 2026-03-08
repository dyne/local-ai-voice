from __future__ import annotations

import pathlib


def test_spec_collects_uvicorn_and_websocket_submodules() -> None:
    spec_text = pathlib.Path("local-ai-voice.spec").read_text(encoding="utf-8")

    assert "collect_submodules('uvicorn')" in spec_text
    assert "collect_submodules('websockets')" in spec_text
    assert "hiddenimports=hiddenimports" in spec_text


def test_make_spec_command_includes_uvicorn_collection() -> None:
    makefile_text = pathlib.Path("GNUmakefile").read_text(encoding="utf-8")

    assert "--hidden-import uvicorn" in makefile_text
    assert "--collect-submodules uvicorn" in makefile_text
    assert "--collect-submodules websockets" in makefile_text
