from __future__ import annotations

import pathlib
import socket
import time


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def wait_for_server(host: str, port: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for local server on {host}:{port}")


def fallback_url(*, host: str, port: int, tls_certfile: pathlib.Path | None) -> str:
    scheme = "https" if tls_certfile is not None else "http"
    return f"{scheme}://{host}:{port}"


def fallback_message(*, host: str, port: int, tls_certfile: pathlib.Path | None) -> str:
    return f"Desktop UI unavailable. Open {fallback_url(host=host, port=port, tls_certfile=tls_certfile)} in a browser."
