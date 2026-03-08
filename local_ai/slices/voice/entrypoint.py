from __future__ import annotations

import argparse
from collections.abc import Callable


def dispatch_voice_entry(
    *,
    raw_argv: list[str],
    build_help_parser_fn: Callable[[], argparse.ArgumentParser],
    parse_dispatch_args_fn: Callable[[list[str]], tuple[object, list[str]]],
    run_transcribe_fn: Callable[[list[str]], int],
    parse_browser_args_fn: Callable[[list[str]], object],
    run_server_fn: Callable[[object], int],
    run_desktop_fn: Callable[[object], int],
) -> int:
    if any(arg in {"-h", "--help"} for arg in raw_argv) and "--web" not in raw_argv and "--server" not in raw_argv:
        build_help_parser_fn().parse_args(raw_argv)
        return 0

    args, remaining = parse_dispatch_args_fn(raw_argv)

    if getattr(args, "server"):
        return run_server_fn(parse_browser_args_fn(remaining))
    if getattr(args, "web"):
        return run_desktop_fn(parse_browser_args_fn(remaining))
    if getattr(args, "cli"):
        return run_transcribe_fn(remaining)
    if any(not arg.startswith("-") for arg in remaining):
        return run_transcribe_fn(remaining)
    return run_desktop_fn(parse_browser_args_fn(remaining))
