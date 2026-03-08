from __future__ import annotations

import argparse
import io
from dataclasses import dataclass

from local_ai.slices.voice.web_ui.launch_modes import run_desktop_mode, run_server_mode


@dataclass
class FakeContext:
    selected_device: str = "CPU"
    model_dir: str = "model"


class FakeService:
    def __init__(self) -> None:
        self.apps_built = 0

    def build_app(self) -> object:
        self.apps_built += 1
        return {"app": self.apps_built}


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=8000,
        tls_certfile=None,
        tls_keyfile=None,
        verbose=True,
    )


def test_run_server_mode_runs_uvicorn_and_logs() -> None:
    stderr = io.StringIO()
    service = FakeService()
    log_messages: list[tuple[str, bool, float]] = []
    uvicorn_calls: list[dict[str, object]] = []
    network_enabled: list[bool] = []

    class FakeUvicorn:
        @staticmethod
        def run(app: object, **kwargs: object) -> None:
            uvicorn_calls.append({"app": app, **kwargs})

    exit_code = run_server_mode(
        args=make_args(),
        prepare_server_fn=lambda args: (FakeContext(), service, 1.5),
        enable_loopback_only_network_fn=lambda: network_enabled.append(True),
        import_uvicorn_fn=lambda: FakeUvicorn,
        logger=lambda message, verbose, start_time: log_messages.append((message, verbose, start_time)),
        stderr=stderr,
    )

    assert exit_code == 0
    assert network_enabled == [True]
    assert uvicorn_calls == [
        {
            "app": {"app": 1},
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
            "ssl_certfile": None,
            "ssl_keyfile": None,
        }
    ]
    assert log_messages == [("Starting server on http://127.0.0.1:8000", True, 1.5)]
    assert "Using device: CPU" in stderr.getvalue()
    assert "Using model: model" in stderr.getvalue()


def test_run_server_mode_returns_validation_code() -> None:
    stderr = io.StringIO()

    exit_code = run_server_mode(
        args=make_args(),
        prepare_server_fn=lambda args: (_ for _ in ()).throw(ValueError("bad config")),
        enable_loopback_only_network_fn=lambda: None,
        import_uvicorn_fn=lambda: object(),
        logger=lambda message, verbose, start_time: None,
        stderr=stderr,
    )

    assert exit_code == 2
    assert "Error: bad config" in stderr.getvalue()


def test_run_desktop_mode_falls_back_when_desktop_dependencies_are_missing() -> None:
    stderr = io.StringIO()
    fallback_calls: list[object] = []
    server_calls: list[object] = []

    exit_code = run_desktop_mode(
        args=make_args(),
        prepare_server_fn=lambda args: (FakeContext(), FakeService(), 2.0),
        run_server_fn=lambda args: server_calls.append((args.host, args.port)) or 7,
        desktop_host_fn=lambda: "127.0.0.1",
        find_free_port_fn=lambda host: 9001,
        enable_loopback_only_network_fn=lambda: None,
        import_desktop_dependencies_fn=lambda: (_ for _ in ()).throw(ImportError("webview missing")),
        wait_for_server_fn=lambda host, port: None,
        logger=lambda message, verbose, start_time: None,
        print_fallback_url_fn=lambda args: fallback_calls.append((args.host, args.port)),
        thread_factory=lambda **kwargs: None,
        stderr=stderr,
    )

    assert exit_code == 7
    assert fallback_calls == [("127.0.0.1", 9001)]
    assert server_calls == [("127.0.0.1", 9001)]
    assert "Desktop UI unavailable: webview missing" in stderr.getvalue()


def test_run_desktop_mode_stops_server_and_falls_back_when_webview_fails() -> None:
    stderr = io.StringIO()
    fallback_calls: list[object] = []
    server_calls: list[object] = []
    network_enabled: list[bool] = []
    wait_calls: list[tuple[str, int]] = []
    log_messages: list[str] = []

    class FakeServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.should_exit = False
            self.force_exit = False

        def run(self) -> None:
            return None

    class FakeUvicorn:
        class Config:
            def __init__(self, app: object, **kwargs: object) -> None:
                self.app = app
                self.kwargs = kwargs

        Server = FakeServer

    class FakeWebView:
        @staticmethod
        def create_window(*args: object, **kwargs: object) -> None:
            return None

        @staticmethod
        def start() -> None:
            raise RuntimeError("window failed")

    class FakeThread:
        def __init__(self, *, target, daemon: bool) -> None:
            self.target = target
            self.daemon = daemon
            self.started = False
            self.join_timeout: float | None = None

        def start(self) -> None:
            self.started = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    thread_holder: list[FakeThread] = []

    exit_code = run_desktop_mode(
        args=make_args(),
        prepare_server_fn=lambda args: (FakeContext(), FakeService(), 2.0),
        run_server_fn=lambda args: server_calls.append((args.host, args.port)) or 9,
        desktop_host_fn=lambda: "127.0.0.1",
        find_free_port_fn=lambda host: 9002,
        enable_loopback_only_network_fn=lambda: network_enabled.append(True),
        import_desktop_dependencies_fn=lambda: (FakeUvicorn, FakeWebView),
        wait_for_server_fn=lambda host, port: wait_calls.append((host, port)),
        logger=lambda message, verbose, start_time: log_messages.append(message),
        print_fallback_url_fn=lambda args: fallback_calls.append((args.host, args.port)),
        thread_factory=lambda **kwargs: thread_holder.append(FakeThread(**kwargs)) or thread_holder[-1],
        stderr=stderr,
    )

    assert exit_code == 9
    assert network_enabled == [True]
    assert wait_calls == [("127.0.0.1", 9002)]
    assert fallback_calls == [("127.0.0.1", 9002)]
    assert server_calls == [("127.0.0.1", 9002)]
    assert log_messages == ["Starting desktop UI on http://127.0.0.1:9002"]
    assert thread_holder[0].started is True
    assert thread_holder[0].join_timeout == 10.0
    assert "Desktop UI unavailable: window failed" in stderr.getvalue()


def test_run_desktop_mode_returns_zero_on_success() -> None:
    stderr = io.StringIO()
    network_enabled: list[bool] = []
    wait_calls: list[tuple[str, int]] = []
    log_messages: list[str] = []

    class FakeServer:
        def __init__(self, config: object) -> None:
            self.config = config
            self.should_exit = False
            self.force_exit = False

        def run(self) -> None:
            return None

    class FakeUvicorn:
        class Config:
            def __init__(self, app: object, **kwargs: object) -> None:
                self.app = app
                self.kwargs = kwargs

        Server = FakeServer

    class FakeWebView:
        create_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        started = False

        @classmethod
        def create_window(cls, *args: object, **kwargs: object) -> None:
            cls.create_calls.append((args, kwargs))

        @classmethod
        def start(cls) -> None:
            cls.started = True

    class FakeThread:
        def __init__(self, *, target, daemon: bool) -> None:
            self.target = target
            self.daemon = daemon
            self.started = False
            self.join_timeout: float | None = None

        def start(self) -> None:
            self.started = True

        def join(self, timeout: float | None = None) -> None:
            self.join_timeout = timeout

    thread_holder: list[FakeThread] = []

    exit_code = run_desktop_mode(
        args=make_args(),
        prepare_server_fn=lambda args: (FakeContext(), FakeService(), 2.0),
        run_server_fn=lambda args: 99,
        desktop_host_fn=lambda: "127.0.0.1",
        find_free_port_fn=lambda host: 9003,
        enable_loopback_only_network_fn=lambda: network_enabled.append(True),
        import_desktop_dependencies_fn=lambda: (FakeUvicorn, FakeWebView),
        wait_for_server_fn=lambda host, port: wait_calls.append((host, port)),
        logger=lambda message, verbose, start_time: log_messages.append(message),
        print_fallback_url_fn=lambda args: None,
        thread_factory=lambda **kwargs: thread_holder.append(FakeThread(**kwargs)) or thread_holder[-1],
        stderr=stderr,
    )

    assert exit_code == 0
    assert network_enabled == [True]
    assert wait_calls == [("127.0.0.1", 9003)]
    assert log_messages == ["Starting desktop UI on http://127.0.0.1:9003"]
    assert FakeWebView.create_calls == [(("Local AI Voice", "http://127.0.0.1:9003"), {"width": 1280, "height": 900})]
    assert FakeWebView.started is True
    assert thread_holder[0].join_timeout == 10.0
    assert stderr.getvalue().count("Using device: CPU") == 1
