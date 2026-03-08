#!/usr/bin/env python
import argparse
import pathlib
import sys
import time

from local_ai.slices.voice.shared.audio_processing import (
    NR_IMPORT_ERROR,
    create_audio_preprocessor,
    VAD_HANGOVER_MS,
    VAD_MIN_SPEECH_FRAMES,
    VAD_MIN_SPEECH_RATIO,
    VAD_MIN_UTTERANCE_MS,
    VAD_MODE,
)
from local_ai.slices.voice.shared.transcript_policy import (
    setup_error_exit_code,
)
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest
from local_ai.slices.voice.transcribe_file.service import execute_transcribe_file
from local_ai.slices.voice.transcribe_runner import execute_transcribe_args
from local_ai.slices.voice.transcribe_live.request import TranscribeLiveRequest
from local_ai.slices.voice.transcribe_live.service import execute_transcribe_live
from local_ai.slices.voice.entrypoint import dispatch_voice_entry
from local_ai.infrastructure.openvino.runtime_env import configure_openvino_runtime_env
from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai.infrastructure.openvino.whisper import (
    create_whisper_runtime,
)


def log(message: str, verbose: bool, start_time: float | None = None) -> None:
    if not verbose:
        return
    prefix = "[transcribe]" if start_time is None else f"[transcribe t+{time.perf_counter() - start_time:.2f}s]"
    print(f"{prefix} {message}", file=sys.stderr, flush=True)


def fail(reason: str, details: list[str] | None = None, exit_code: int = 1) -> int:
    print(f"Error: {reason}", file=sys.stderr)
    if details:
        for detail in details:
            print(f"- {detail}", file=sys.stderr)
    return exit_code

def run_file_mode(
    args: argparse.Namespace,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
) -> int:
    if args.wav_path is None:
        return fail("Internal error: file mode requires wav_path.", exit_code=9)
    response = execute_transcribe_file(
        request=TranscribeFileRequest(wav_path=args.wav_path, verbose=args.verbose),
        pipe=pipe,
        audio_preprocessor=audio_preprocessor,
        generate_kwargs=generate_kwargs,
        start=start,
        logger=log,
        runtime_error_details=likely_reason_details,
    )
    if response.exit_code != 0:
        return fail(response.reason or "Transcription failed.", response.details, exit_code=response.exit_code)
    print(response.text or "")
    return response.exit_code


def run_live_mode(
    args: argparse.Namespace,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
) -> int:
    try:
        import sounddevice as sd
    except Exception as exc:
        sd = exc

    response = execute_transcribe_live(
        request=TranscribeLiveRequest(
            chunk_seconds=args.chunk_seconds,
            silence_detect=args.silence_detect,
            verbose=args.verbose,
        ),
        sounddevice_module=sd,
        pipe=pipe,
        audio_preprocessor=audio_preprocessor,
        generate_kwargs=generate_kwargs,
        start=start,
        logger=log,
        runtime_error_details=likely_reason_details,
        on_output=lambda line: print(line, flush=True),
        on_status=lambda line: print(line, file=sys.stderr, flush=True),
    )
    if response.exit_code != 0:
        return fail(response.reason or "Live transcription failed.", response.details, exit_code=response.exit_code)
    return response.exit_code


def build_transcribe_parser(*, include_web_flag: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe with OpenVINO GenAI Whisper on NPU/GPU/CPU (file or live microphone mode)."
    )
    if include_web_flag:
        parser.add_argument(
            "--web",
            action="store_true",
            help="Open the desktop web UI instead of file/live transcription.",
        )
        parser.add_argument(
            "--server",
            action="store_true",
            help="Run the browser audio streaming transcription server without the desktop wrapper.",
        )
        parser.add_argument(
            "--cli",
            action="store_true",
            help="Force the non-web CLI mode for file or live microphone transcription.",
        )
    parser.add_argument("wav_path", type=pathlib.Path, nargs="?", help="Optional input .wav path.")
    parser.add_argument(
        "--device",
        default="NPU,GPU,CPU",
        help="Device preference order using NPU,GPU,CPU, or 'list' to print detected devices (default: NPU,GPU,CPU).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional OpenVINO model directory or Hugging Face repo id. If omitted, default OpenVINO model is auto-downloaded.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable model downloads. Fail if required model is not available locally.",
    )
    parser.add_argument("--language", default=None, help="Optional language token like <|en|>.")
    parser.add_argument("--task", default=None, choices=["transcribe", "translate"], help="Optional Whisper task.")
    parser.add_argument("--timestamps", action="store_true", help="Request timestamps in result object.")
    silence_group = parser.add_mutually_exclusive_group()
    silence_group.add_argument(
        "--silence-detect",
        dest="silence_detect",
        action="store_true",
        help="Enable noise reduction and WebRTC VAD speech gating before transcription.",
    )
    silence_group.add_argument(
        "--no-silence-detect",
        dest="silence_detect",
        action="store_false",
        help="Disable noise reduction and WebRTC VAD speech gating.",
    )
    parser.set_defaults(silence_detect=True)
    parser.add_argument("--vad-mode", type=int, choices=[0, 1, 2, 3], default=VAD_MODE, help=f"WebRTC VAD aggressiveness mode (default: {VAD_MODE}).")
    parser.add_argument("--vad-min-speech-frames", type=int, default=VAD_MIN_SPEECH_FRAMES, help=f"Minimum consecutive speech frames required to trigger speech (default: {VAD_MIN_SPEECH_FRAMES}).")
    parser.add_argument("--vad-min-speech-ratio", type=float, default=VAD_MIN_SPEECH_RATIO, help=f"Minimum speech frame ratio per chunk (default: {VAD_MIN_SPEECH_RATIO}).")
    parser.add_argument("--vad-min-utterance-ms", type=int, default=VAD_MIN_UTTERANCE_MS, help=f"Minimum detected speech duration in milliseconds (default: {VAD_MIN_UTTERANCE_MS}).")
    parser.add_argument("--vad-hangover-ms", type=int, default=VAD_HANGOVER_MS, help=f"Hangover duration in milliseconds after speech ends (default: {VAD_HANGOVER_MS}).")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Live mode chunk duration in seconds (used when wav_path is omitted).",
    )
    parser.add_argument("--profile", action="store_true", help="Enable py-spy profiling for this run.")
    parser.add_argument(
        "--profile-output",
        type=pathlib.Path,
        default=None,
        help="Optional py-spy output SVG path (default: profiles/<timestamp>.svg).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")
    return parser


def parse_transcribe_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_transcribe_parser()
    return parser.parse_args(argv)


def run_transcribe(argv: list[str] | None = None) -> int:
    args = parse_transcribe_args(argv)
    profile_session = start_py_spy_profile(
        enabled=args.profile,
        label="local-ai-voice",
        output_path=args.profile_output,
    )
    try:
        return execute_transcribe_args(
            args=args,
            perf_counter_fn=time.perf_counter,
            configure_runtime_env_fn=configure_openvino_runtime_env,
            create_runtime_fn=create_whisper_runtime,
            create_audio_preprocessor_fn=create_audio_preprocessor,
            enable_loopback_only_network_fn=enable_loopback_only_network,
            run_file_mode_fn=run_file_mode,
            run_live_mode_fn=run_live_mode,
            logger=log,
            fail_fn=fail,
            setup_error_exit_code_fn=setup_error_exit_code,
            nr_import_error=NR_IMPORT_ERROR,
            base_dir=pathlib.Path(__file__).resolve().parent,
            stderr=sys.stderr,
        )
    finally:
        stop_py_spy_profile(profile_session)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    
    def parse_dispatch_args(current_argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--web",
            action="store_true",
            help="Open the desktop web UI instead of file/live transcription.",
        )
        parser.add_argument(
            "--server",
            action="store_true",
            help="Run the browser audio streaming transcription server without the desktop wrapper.",
        )
        parser.add_argument(
            "--cli",
            action="store_true",
            help="Force the non-web CLI mode for file or live microphone transcription.",
        )
        return parser.parse_known_args(current_argv)

    return dispatch_voice_entry(
        raw_argv=raw_argv,
        build_help_parser_fn=lambda: build_transcribe_parser(include_web_flag=True),
        parse_dispatch_args_fn=parse_dispatch_args,
        run_transcribe_fn=run_transcribe,
        parse_browser_args_fn=lambda remaining: __import__("browser_webrtc").parse_args(remaining),
        run_server_fn=lambda args: __import__("browser_webrtc").run_server(args),
        run_desktop_fn=lambda args: __import__("browser_webrtc").run_desktop(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
