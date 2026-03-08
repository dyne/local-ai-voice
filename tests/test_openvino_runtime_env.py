from __future__ import annotations

import os
import pathlib
import sys

from local_ai.infrastructure.openvino.runtime_env import configure_openvino_runtime_env


def test_configure_openvino_runtime_env_noops_when_not_frozen(monkeypatch) -> None:
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.delenv("OPENVINO_LIB_PATHS", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    configure_openvino_runtime_env()

    assert os.environ.get("OPENVINO_LIB_PATHS") is None


def test_configure_openvino_runtime_env_prepends_existing_paths(tmp_path: pathlib.Path, monkeypatch) -> None:
    meipass = tmp_path / "bundle"
    libs = meipass / "openvino" / "libs"
    tokenizers = meipass / "openvino_tokenizers" / "libs"
    exe_dir = tmp_path / "dist"
    exe_libs = exe_dir / "openvino_genai"
    libs.mkdir(parents=True)
    tokenizers.mkdir(parents=True)
    exe_libs.mkdir(parents=True)
    fake_exe = exe_dir / "local-ai-voice.exe"
    fake_exe.write_text("x", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe), raising=False)
    monkeypatch.setenv("PATH", "existing-path")
    monkeypatch.setenv("OPENVINO_LIB_PATHS", "existing-lib-path")

    configure_openvino_runtime_env()

    path_parts = os.environ["PATH"].split(os.pathsep)
    lib_parts = os.environ["OPENVINO_LIB_PATHS"].split(os.pathsep)
    assert str(libs) in path_parts
    assert str(tokenizers) in path_parts
    assert str(exe_libs) in path_parts
    assert path_parts[-1] == "existing-path"
    assert lib_parts[-1] == "existing-lib-path"
