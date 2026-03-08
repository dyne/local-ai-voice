from __future__ import annotations

import argparse

from local_ai.slices.voice.entrypoint import dispatch_voice_entry


def parse_dispatch_args(raw_argv: list[str]) -> tuple[object, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--cli", action="store_true")
    return parser.parse_known_args(raw_argv)


def test_dispatch_voice_entry_uses_help_parser_for_help() -> None:
    help_calls: list[list[str]] = []

    class FakeParser:
        def parse_args(self, raw_argv: list[str]) -> object:
            help_calls.append(raw_argv)
            return object()

    result = dispatch_voice_entry(
        raw_argv=["--help"],
        build_help_parser_fn=lambda: FakeParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: 1,
        parse_browser_args_fn=lambda argv: argv,
        run_server_fn=lambda args: 2,
        run_desktop_fn=lambda args: 3,
    )

    assert result == 0
    assert help_calls == [["--help"]]


def test_dispatch_voice_entry_routes_server_mode() -> None:
    browser_args: list[list[str]] = []

    result = dispatch_voice_entry(
        raw_argv=["--server", "--port", "9000"],
        build_help_parser_fn=lambda: argparse.ArgumentParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: 1,
        parse_browser_args_fn=lambda argv: browser_args.append(argv) or {"argv": argv},
        run_server_fn=lambda args: 7,
        run_desktop_fn=lambda args: 3,
    )

    assert result == 7
    assert browser_args == [["--port", "9000"]]


def test_dispatch_voice_entry_routes_web_mode() -> None:
    browser_args: list[list[str]] = []

    result = dispatch_voice_entry(
        raw_argv=["--web", "--verbose"],
        build_help_parser_fn=lambda: argparse.ArgumentParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: 1,
        parse_browser_args_fn=lambda argv: browser_args.append(argv) or {"argv": argv},
        run_server_fn=lambda args: 2,
        run_desktop_fn=lambda args: 8,
    )

    assert result == 8
    assert browser_args == [["--verbose"]]


def test_dispatch_voice_entry_routes_cli_mode() -> None:
    transcribe_calls: list[list[str]] = []

    result = dispatch_voice_entry(
        raw_argv=["--cli", "--device", "CPU"],
        build_help_parser_fn=lambda: argparse.ArgumentParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: transcribe_calls.append(argv) or 4,
        parse_browser_args_fn=lambda argv: {"argv": argv},
        run_server_fn=lambda args: 2,
        run_desktop_fn=lambda args: 3,
    )

    assert result == 4
    assert transcribe_calls == [["--device", "CPU"]]


def test_dispatch_voice_entry_routes_file_argument_to_cli() -> None:
    transcribe_calls: list[list[str]] = []

    result = dispatch_voice_entry(
        raw_argv=["sample.wav", "--device", "CPU"],
        build_help_parser_fn=lambda: argparse.ArgumentParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: transcribe_calls.append(argv) or 5,
        parse_browser_args_fn=lambda argv: {"argv": argv},
        run_server_fn=lambda args: 2,
        run_desktop_fn=lambda args: 3,
    )

    assert result == 5
    assert transcribe_calls == [["sample.wav", "--device", "CPU"]]


def test_dispatch_voice_entry_defaults_to_desktop_mode() -> None:
    browser_args: list[list[str]] = []

    result = dispatch_voice_entry(
        raw_argv=[],
        build_help_parser_fn=lambda: argparse.ArgumentParser(),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=lambda argv: 1,
        parse_browser_args_fn=lambda argv: browser_args.append(argv) or {"argv": argv},
        run_server_fn=lambda args: 2,
        run_desktop_fn=lambda args: 9,
    )

    assert result == 9
    assert browser_args == [[]]
