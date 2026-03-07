from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class PySpySession:
    process: subprocess.Popen[bytes]
    output_path: pathlib.Path


def start_py_spy_profile(
    *,
    enabled: bool,
    label: str,
    output_path: pathlib.Path | None = None,
    sample_rate: int = 100,
) -> PySpySession | None:
    if not enabled:
        return None

    exe = shutil.which("py-spy")
    if exe is None:
        print("Warning: --profile requested but py-spy is not installed or not on PATH.", file=sys.stderr, flush=True)
        return None

    if output_path is None:
        out_dir = pathlib.Path.cwd() / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.svg"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "record",
        "--pid",
        str(os.getpid()),
        "--rate",
        str(sample_rate),
        "--output",
        str(output_path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"Warning: failed to start py-spy profiler: {exc}", file=sys.stderr, flush=True)
        return None

    print(f"[profile] py-spy recording to {output_path}", file=sys.stderr, flush=True)
    return PySpySession(process=proc, output_path=output_path)


def stop_py_spy_profile(session: PySpySession | None) -> None:
    if session is None:
        return

    proc = session.process
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
    print(f"[profile] py-spy output saved to {session.output_path}", file=sys.stderr, flush=True)
