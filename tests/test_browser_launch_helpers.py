from __future__ import annotations

import pathlib

import pytest

from local_ai.slices.voice.web_ui.launch_helpers import fallback_message, fallback_url, find_free_port, wait_for_server


def test_fallback_url_uses_http_without_tls(tmp_path: pathlib.Path) -> None:
    url = fallback_url(host="127.0.0.1", port=8000, tls_certfile=None)
    assert url == "http://127.0.0.1:8000"


def test_fallback_url_uses_https_with_tls(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("x", encoding="utf-8")
    url = fallback_url(host="127.0.0.1", port=8443, tls_certfile=cert)
    assert url == "https://127.0.0.1:8443"


def test_wait_for_server_returns_when_connection_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        "local_ai.slices.voice.web_ui.launch_helpers.socket.create_connection",
        lambda *args, **kwargs: FakeConnection(),
    )

    wait_for_server("127.0.0.1", 8000, timeout_seconds=0.01)


def test_wait_for_server_retries_until_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    time_values = iter([0.0, 0.02, 0.04, 0.06])
    monkeypatch.setattr(
        "local_ai.slices.voice.web_ui.launch_helpers.socket.create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("down")),
    )
    monkeypatch.setattr("local_ai.slices.voice.web_ui.launch_helpers.time.monotonic", lambda: next(time_values))
    monkeypatch.setattr("local_ai.slices.voice.web_ui.launch_helpers.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Timed out waiting for local server"):
        wait_for_server("127.0.0.1", 8000, timeout_seconds=0.05)


def test_find_free_port_returns_bound_port(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def bind(self, address) -> None:
            self.address = address

        def listen(self, backlog: int) -> None:
            self.backlog = backlog

        def getsockname(self):
            return ("127.0.0.1", 4321)

    monkeypatch.setattr("local_ai.slices.voice.web_ui.launch_helpers.socket.socket", lambda *args, **kwargs: FakeSocket())

    assert find_free_port("127.0.0.1") == 4321


def test_fallback_message_uses_resolved_url(tmp_path: pathlib.Path) -> None:
    cert = tmp_path / "cert.pem"
    cert.write_text("x", encoding="utf-8")
    message = fallback_message(host="127.0.0.1", port=8443, tls_certfile=cert)
    assert message == "Desktop UI unavailable. Open https://127.0.0.1:8443 in a browser."
