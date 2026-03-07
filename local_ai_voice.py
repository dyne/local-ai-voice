#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import pathlib


def _load_impl() -> object:
    impl_path = pathlib.Path(__file__).resolve().with_name("local-ai-voice.py")
    spec = importlib.util.spec_from_file_location("local_ai_voice_impl", impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load implementation module: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = _load_impl()
for _name, _value in vars(_IMPL).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(main())
