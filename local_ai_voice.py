#!/usr/bin/env python
import argparse
import os
import pathlib
import sys
import time

import numpy as np

from local_ai.slices.voice.shared.audio_processing import (
    NR_IMPORT_ERROR,
    TARGET_SAMPLE_RATE,
    create_audio_preprocessor,
    ensure_sample_rate,
    normalize_audio_format,
    preprocess_audio,
    read_wav_mono_float32,
    resample_audio_linear,
    VAD_HANGOVER_MS,
    VAD_MIN_SPEECH_FRAMES,
    VAD_MIN_SPEECH_RATIO,
    VAD_MIN_UTTERANCE_MS,
    VAD_MODE,
)
from local_ai.slices.voice.shared.transcript_policy import (
    setup_error_exit_code,
    should_suppress_transcript,
    transcribe_chunk,
)
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest
from local_ai.slices.voice.transcribe_file.service import execute_transcribe_file
from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from local_ai.infrastructure.openvino.whisper import (
    create_whisper_runtime,
    likely_reason_details,
)
from local_ai.shared.domain.errors import (
    DeviceListRequested,
    PipelineSetupError,
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
    if args.chunk_seconds <= 0:
        return fail("--chunk-seconds must be > 0.", exit_code=2)

    try:
        import sounddevice as sd
    except Exception as exc:
        return fail(
            "Live mode requires sounddevice.",
            [f"Import error: {exc}", "Install sounddevice or pass a WAV file instead."],
            exit_code=6,
        )

    try:
        default_input = sd.query_devices(kind="input")
        record_sample_rate = int(round(float(default_input["default_samplerate"])))
    except Exception as exc:
        return fail(
            "Could not read default input sample rate.",
            [f"Runtime error: {exc}", "Check your default microphone device settings."],
            exit_code=7,
        )

    if record_sample_rate <= 0:
        return fail("Default input sample rate is invalid.", [f"Detected: {record_sample_rate}"], exit_code=7)

    chunk_samples = int(round(args.chunk_seconds * record_sample_rate))
    if chunk_samples <= 0:
        return fail("Computed chunk size is invalid; increase --chunk-seconds.", exit_code=2)

    print("Live transcription started. Press Ctrl+C to stop.", file=sys.stderr, flush=True)
    log(
        f"Recording {record_sample_rate} Hz mono; transcribing every {args.chunk_seconds:.2f}s.",
        args.verbose,
        start,
    )

    try:
        with sd.InputStream(
            samplerate=record_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            chunk_index = 0
            while True:
                data, overflowed = stream.read(chunk_samples)
                chunk_index += 1
                if overflowed:
                    log("Audio input overflow detected; transcription may skip samples.", args.verbose, start)

                audio = np.asarray(data[:, 0], dtype=np.float32)
                if np.max(np.abs(audio), initial=0.0) < 1e-4:
                    continue
                try:
                    audio = preprocess_audio(audio, record_sample_rate, audio_preprocessor, args.verbose, start, log)
                except Exception as exc:
                    return fail(
                        "Live audio preprocessing failed.",
                        [f"Runtime error: {exc}", NR_IMPORT_ERROR],
                        exit_code=6,
                    )
                if args.silence_detect and audio.size == 0:
                    continue
                if record_sample_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio_linear(audio, record_sample_rate, TARGET_SAMPLE_RATE)

                try:
                    text = transcribe_chunk(pipe, audio, generate_kwargs)
                except Exception as exc:
                    return fail("Live transcription failed.", likely_reason_details(exc), exit_code=5)

                if text and not should_suppress_transcript(text, audio_preprocessor):
                    print(f"[chunk {chunk_index}] {text}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped live transcription.", file=sys.stderr, flush=True)
        return 0
    except Exception as exc:
        return fail(
            "Failed to capture microphone audio.",
            [f"Runtime error: {exc}", "Check microphone permissions and input device availability."],
            exit_code=7,
        )


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
        start = time.perf_counter()
        configure_openvino_runtime_env()

        try:
            runtime = create_whisper_runtime(
                args=args,
                base_dir=pathlib.Path(__file__).resolve().parent,
                logger=log,
                verbose=args.verbose,
                start_time=start,
            )
        except DeviceListRequested as exc:
            if not exc.devices:
                print("No OpenVINO devices detected")
                return 1
            for dev in exc.devices:
                print(dev)
            return 0
        except PipelineSetupError as exc:
            return fail(exc.reason, exc.details, exit_code=setup_error_exit_code(exc.reason))

        print(f"Using device: {runtime.selected_device}", file=sys.stderr, flush=True)
        print(f"Using model: {runtime.model_dir}", file=sys.stderr, flush=True)
        enable_loopback_only_network()

        pipe = runtime.pipe
        try:
            audio_preprocessor = create_audio_preprocessor(
                args.silence_detect,
                vad_mode=args.vad_mode,
                min_speech_frames=args.vad_min_speech_frames,
                min_speech_ratio=args.vad_min_speech_ratio,
                min_utterance_ms=args.vad_min_utterance_ms,
                hangover_ms=args.vad_hangover_ms,
            )
        except Exception as exc:
            return fail("Audio preprocessing failed.", [f"Runtime error: {exc}", NR_IMPORT_ERROR], exit_code=6)
        generate_kwargs = runtime.generate_kwargs
        status = (
            run_file_mode(args, pipe, audio_preprocessor, generate_kwargs, start)
            if args.wav_path
            else run_live_mode(args, pipe, audio_preprocessor, generate_kwargs, start)
        )
        if status == 0:
            log(f"Done in {time.perf_counter() - start:.2f}s", args.verbose, start)
        return status
    finally:
        stop_py_spy_profile(profile_session)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in raw_argv) and "--web" not in raw_argv and "--server" not in raw_argv:
        build_transcribe_parser(include_web_flag=True).parse_args(raw_argv)
        return 0
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
    args, remaining = parser.parse_known_args(raw_argv)

    if args.server:
        import browser_webrtc

        return browser_webrtc.run_server(browser_webrtc.parse_args(remaining))
    if args.web:
        import browser_webrtc

        return browser_webrtc.run_desktop(browser_webrtc.parse_args(remaining))
    if args.cli:
        return run_transcribe(remaining)
    if any(not arg.startswith("-") for arg in remaining):
        return run_transcribe(remaining)
    if remaining:
        import browser_webrtc

        return browser_webrtc.run_desktop(browser_webrtc.parse_args(remaining))
    import browser_webrtc

    return browser_webrtc.run_desktop(browser_webrtc.parse_args(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
