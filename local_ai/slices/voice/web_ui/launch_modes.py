from __future__ import annotations

from collections.abc import Callable
from typing import TextIO


def _print_runtime_selection(ctx: object, stderr: TextIO) -> None:
    print(f"Using device: {getattr(ctx, 'selected_device')}", file=stderr, flush=True)
    print(f"Using model: {getattr(ctx, 'model_dir')}", file=stderr, flush=True)


def run_server_mode(
    *,
    args: object,
    prepare_server_fn: Callable[[object], tuple[object, object, float]],
    enable_loopback_only_network_fn: Callable[[], None],
    import_uvicorn_fn: Callable[[], object],
    logger: Callable[[str, bool, float | None], None],
    stderr: TextIO,
) -> int:
    try:
        ctx, service, start_time = prepare_server_fn(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=stderr)
        return 3

    _print_runtime_selection(ctx, stderr)
    enable_loopback_only_network_fn()

    try:
        uvicorn = import_uvicorn_fn()
    except Exception as exc:
        print(f"Error: uvicorn is not available: {exc}", file=stderr)
        return 3

    scheme = "https" if getattr(args, "tls_certfile") is not None else "http"
    logger(f"Starting server on {scheme}://{getattr(args, 'host')}:{getattr(args, 'port')}", getattr(args, "verbose"), start_time)
    uvicorn.run(
        service.build_app(),
        host=getattr(args, "host"),
        port=getattr(args, "port"),
        log_level="info",
        ssl_certfile=str(getattr(args, "tls_certfile")) if getattr(args, "tls_certfile") is not None else None,
        ssl_keyfile=str(getattr(args, "tls_keyfile")) if getattr(args, "tls_keyfile") is not None else None,
    )
    return 0


def run_desktop_mode(
    *,
    args: object,
    prepare_server_fn: Callable[[object], tuple[object, object, float]],
    run_server_fn: Callable[[object], int],
    desktop_host_fn: Callable[[], str],
    find_free_port_fn: Callable[[str], int],
    enable_loopback_only_network_fn: Callable[[], None],
    import_desktop_dependencies_fn: Callable[[], tuple[object, object]],
    wait_for_server_fn: Callable[[str, int], None],
    logger: Callable[[str, bool, float | None], None],
    print_fallback_url_fn: Callable[[object], None],
    thread_factory: Callable[..., object],
    stderr: TextIO,
) -> int:
    try:
        setattr(args, "host", desktop_host_fn())
        if getattr(args, "port") == 8000:
            setattr(args, "port", find_free_port_fn(getattr(args, "host")))
        ctx, service, start_time = prepare_server_fn(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=stderr)
        return 2
    except RuntimeError as exc:
        print(f"Error: {exc}", file=stderr)
        return 3

    _print_runtime_selection(ctx, stderr)

    try:
        uvicorn, webview = import_desktop_dependencies_fn()
    except Exception as exc:
        print(f"Desktop UI unavailable: {exc}", file=stderr, flush=True)
        print_fallback_url_fn(args)
        return run_server_fn(args)

    config = uvicorn.Config(
        service.build_app(),
        host=getattr(args, "host"),
        port=getattr(args, "port"),
        log_level="info",
        ssl_certfile=str(getattr(args, "tls_certfile")) if getattr(args, "tls_certfile") is not None else None,
        ssl_keyfile=str(getattr(args, "tls_keyfile")) if getattr(args, "tls_keyfile") is not None else None,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    server_thread = thread_factory(target=server.run, daemon=True)
    server_thread.start()
    wait_for_server_fn(getattr(args, "host"), getattr(args, "port"))
    enable_loopback_only_network_fn()

    scheme = "https" if getattr(args, "tls_certfile") is not None else "http"
    url = f"{scheme}://{getattr(args, 'host')}:{getattr(args, 'port')}"
    logger(f"Starting desktop UI on {url}", getattr(args, "verbose"), start_time)
    try:
        webview.create_window("Local AI Voice", url, width=1280, height=900)
        webview.start()
    except Exception as exc:
        print(f"Desktop UI unavailable: {exc}", file=stderr, flush=True)
        print_fallback_url_fn(args)
        server.should_exit = True
        server_thread.join(timeout=10.0)
        return run_server_fn(args)

    server.should_exit = True
    server_thread.join(timeout=10.0)
    if server.force_exit:
        return 3
    return 0
