from __future__ import annotations

import pathlib


def test_spec_collects_uvicorn_and_websocket_submodules() -> None:
    spec_text = pathlib.Path("local-ai-voice.spec").read_text(encoding="utf-8")

    assert "collect_submodules('uvicorn')" in spec_text
    assert "collect_submodules('websockets')" in spec_text
    assert "hiddenimports=hiddenimports" in spec_text
    assert "('frontend/dist', 'frontend/dist')" in spec_text


def test_make_spec_command_includes_uvicorn_collection() -> None:
    makefile_text = pathlib.Path("GNUmakefile").read_text(encoding="utf-8")

    assert "--hidden-import uvicorn" in makefile_text
    assert "--collect-submodules uvicorn" in makefile_text
    assert "--collect-submodules websockets" in makefile_text
    assert "frontend-build" in makefile_text
    assert '--add-data "$(FRONTEND_DIR)/dist$(DATA_SEP)$(FRONTEND_DIR)/dist"' in makefile_text


def test_github_release_workflow_includes_uvicorn_collection() -> None:
    workflow_text = pathlib.Path(".github/workflows/build-and-release.yaml").read_text(
        encoding="utf-8"
    )

    assert "--hidden-import uvicorn" in workflow_text
    assert "--collect-submodules uvicorn" in workflow_text
    assert "--collect-submodules websockets" in workflow_text
    assert "actions/setup-node@v4" in workflow_text
    assert "npm --prefix frontend test" in workflow_text
    assert "npm --prefix frontend run build" in workflow_text
    assert '--add-data "frontend/dist;frontend/dist"' in workflow_text
    assert '--add-data "frontend/dist:frontend/dist"' in workflow_text
