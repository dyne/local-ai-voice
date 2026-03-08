from __future__ import annotations

import os
import pathlib
import sys


def configure_openvino_runtime_env() -> None:
    candidates: list[pathlib.Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            base = pathlib.Path(meipass)
            candidates.extend(
                [
                    base / "openvino" / "libs",
                    base / "openvino_genai",
                    base / "openvino_tokenizers",
                    base / "openvino_tokenizers" / "libs",
                    base,
                ]
            )
        exe_dir = pathlib.Path(sys.executable).resolve().parent
        candidates.extend(
            [
                exe_dir / "openvino" / "libs",
                exe_dir / "openvino_genai",
                exe_dir / "openvino_tokenizers",
                exe_dir / "openvino_tokenizers" / "libs",
                exe_dir,
            ]
        )

    valid = [p for p in candidates if p.exists()]
    if not valid:
        return

    path_sep = os.pathsep
    existing_path = os.environ.get("PATH", "")
    prepend = [str(p) for p in valid]
    os.environ["PATH"] = path_sep.join(prepend + ([existing_path] if existing_path else []))

    existing_lib_paths = os.environ.get("OPENVINO_LIB_PATHS", "")
    merged = prepend + ([existing_lib_paths] if existing_lib_paths else [])
    os.environ["OPENVINO_LIB_PATHS"] = path_sep.join(merged)
