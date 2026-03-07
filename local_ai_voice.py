#!/usr/bin/env python
import argparse
import os
import pathlib
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np

from network_guard import enable_loopback_only_network
from pyspy_profile import start_py_spy_profile, stop_py_spy_profile
from voice_runtime import (
    DeviceListRequested,
    PipelineSetupError,
    create_whisper_runtime,
    likely_reason_details,
)

TARGET_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_MODE = 2
VAD_MIN_SPEECH_FRAMES = 3
VAD_MIN_SPEECH_RATIO = 0.2
VAD_MIN_UTTERANCE_MS = 180
VAD_HANGOVER_MS = 300
NR_IMPORT_ERROR = "Install noisereduce and webrtcvad-wheels."


@dataclass
class AudioPreprocessor:
    nr: object
    vad: object
    vad_mode: int
    min_speech_frames: int
    min_speech_ratio: float
    min_utterance_ms: int
    hangover_ms: int
    hangover_frames: int = 0


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


def read_wav_mono_float32(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


def resample_audio_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Invalid sample rate(s): source={source_rate}, target={target_rate}")
    if audio.size == 0:
        return audio

    duration = audio.shape[0] / float(source_rate)
    out_length = max(1, int(round(duration * target_rate)))
    in_times = np.arange(audio.shape[0], dtype=np.float64) / float(source_rate)
    out_times = np.arange(out_length, dtype=np.float64) / float(target_rate)
    return np.interp(out_times, in_times, audio).astype(np.float32)


def ensure_sample_rate(audio: np.ndarray, sample_rate: int, verbose: bool, start: float) -> tuple[np.ndarray, int]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio, sample_rate
    log(f"Resampling audio from {sample_rate} Hz to {TARGET_SAMPLE_RATE} Hz", verbose, start)
    return resample_audio_linear(audio, sample_rate, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE


def create_audio_preprocessor(
    enabled: bool,
    *,
    vad_mode: int = VAD_MODE,
    min_speech_frames: int = VAD_MIN_SPEECH_FRAMES,
    min_speech_ratio: float = VAD_MIN_SPEECH_RATIO,
    min_utterance_ms: int = VAD_MIN_UTTERANCE_MS,
    hangover_ms: int = VAD_HANGOVER_MS,
) -> object | None:
    if not enabled:
        return None
    if vad_mode not in (0, 1, 2, 3):
        raise RuntimeError(f"Invalid VAD mode: {vad_mode}. Use 0, 1, 2, or 3.")
    if min_speech_frames <= 0:
        raise RuntimeError("min_speech_frames must be > 0.")
    if not (0.0 < min_speech_ratio <= 1.0):
        raise RuntimeError("min_speech_ratio must be > 0 and <= 1.")
    if min_utterance_ms <= 0:
        raise RuntimeError("min_utterance_ms must be > 0.")
    if hangover_ms < 0:
        raise RuntimeError("hangover_ms must be >= 0.")
    try:
        import noisereduce as nr
        import webrtcvad
    except Exception as exc:
        raise RuntimeError("Failed to import audio preprocessing dependencies.") from exc
    return AudioPreprocessor(
        nr=nr,
        vad=webrtcvad.Vad(vad_mode),
        vad_mode=vad_mode,
        min_speech_frames=min_speech_frames,
        min_speech_ratio=min_speech_ratio,
        min_utterance_ms=min_utterance_ms,
        hangover_ms=hangover_ms,
    )


def speech_frame_stats(audio: np.ndarray, vad: object) -> tuple[int, int, int]:
    frame_samples = TARGET_SAMPLE_RATE * VAD_FRAME_MS // 1000
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    if pcm16.size == 0:
        return 0, 0, 0
    speech_frames = 0
    max_consecutive = 0
    consecutive = 0
    total_frames = 0
    for start_idx in range(0, pcm16.shape[0], frame_samples):
        frame = pcm16[start_idx:start_idx + frame_samples]
        if frame.size == 0:
            continue
        if frame.size < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.size))
        total_frames += 1
        if vad.is_speech(frame.tobytes(), TARGET_SAMPLE_RATE):
            speech_frames += 1
            consecutive += 1
            if consecutive > max_consecutive:
                max_consecutive = consecutive
        else:
            consecutive = 0
    return speech_frames, total_frames, max_consecutive


def _speech_detected(
    original_audio: np.ndarray,
    reduced_audio: np.ndarray,
    preprocessor: AudioPreprocessor,
) -> bool:
    min_duration_frames = max(1, int(np.ceil(preprocessor.min_utterance_ms / float(VAD_FRAME_MS))))
    hangover_frames = max(1, int(np.ceil(preprocessor.hangover_ms / float(VAD_FRAME_MS))))

    orig_speech, orig_total, orig_run = speech_frame_stats(original_audio, preprocessor.vad)
    red_speech, red_total, red_run = speech_frame_stats(reduced_audio, preprocessor.vad)
    total_frames = max(orig_total, red_total)
    if total_frames == 0:
        preprocessor.hangover_frames = 0
        return False

    speech_frames = max(orig_speech, red_speech)
    speech_ratio = speech_frames / float(total_frames)
    consecutive_run = max(orig_run, red_run)
    speech_now = (
        consecutive_run >= preprocessor.min_speech_frames
        and speech_frames >= min_duration_frames
        and speech_ratio >= preprocessor.min_speech_ratio
    )
    if speech_now:
        preprocessor.hangover_frames = hangover_frames
        return True
    if preprocessor.hangover_frames > 0:
        preprocessor.hangover_frames -= min(total_frames, preprocessor.hangover_frames)
        return True
    return False


def preprocess_audio(
    audio: np.ndarray,
    preprocessor: object | None,
    verbose: bool,
    start: float,
) -> np.ndarray:
    if preprocessor is None:
        return np.asarray(audio, dtype=np.float32, copy=False)
    original_audio = np.asarray(audio, dtype=np.float32, copy=False)
    try:
        reduced = preprocessor.nr.reduce_noise(y=original_audio, sr=TARGET_SAMPLE_RATE)
    except Exception as exc:
        raise RuntimeError("Noise reduction failed.") from exc

    reduced_audio = np.asarray(reduced, dtype=np.float32).reshape(-1)
    if reduced_audio.size == 0:
        return reduced_audio
    if not _speech_detected(original_audio, reduced_audio, preprocessor):
        log("Audio skipped: no speech detected by VAD", verbose, start)
        return np.asarray([], dtype=np.float32)
    return reduced_audio


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


def result_to_text(result: object) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    texts = getattr(result, "texts", None)
    if isinstance(texts, list) and texts:
        return str(texts[0])
    return str(result)


def transcribe_chunk(pipe: object, audio: np.ndarray, generate_kwargs: dict[str, object]) -> str:
    result = pipe.generate(audio.tolist(), **generate_kwargs)
    return result_to_text(result).strip()


def setup_error_exit_code(reason: str) -> int:
    if reason.startswith("Model directory not found:"):
        return 2
    if reason.startswith("Failed to create WhisperPipeline on "):
        return 4
    return 3


def run_file_mode(
    args: argparse.Namespace,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
) -> int:
    if args.wav_path is None:
        return fail("Internal error: file mode requires wav_path.", exit_code=9)
    if not args.wav_path.exists():
        return fail(f"Input file not found: {args.wav_path}", exit_code=2)

    log("Reading WAV", args.verbose, start)
    audio, sample_rate = read_wav_mono_float32(args.wav_path)
    audio, sample_rate = ensure_sample_rate(audio, sample_rate, args.verbose, start)
    try:
        audio = preprocess_audio(audio, audio_preprocessor, args.verbose, start)
    except Exception as exc:
        return fail(
            "Audio preprocessing failed.",
            [f"Runtime error: {exc}", NR_IMPORT_ERROR],
            exit_code=6,
        )
    if audio.size == 0:
        return fail("No speech detected.", ["Input appears silent after noise reduction."], exit_code=2)
    log(f"Audio prepared: samples={audio.shape[0]}, sample_rate={sample_rate} Hz", args.verbose, start)

    try:
        text = transcribe_chunk(pipe, audio, generate_kwargs)
    except Exception as exc:
        return fail("Transcription failed.", likely_reason_details(exc), exit_code=5)

    print(text)
    return 0


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
                if record_sample_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio_linear(audio, record_sample_rate, TARGET_SAMPLE_RATE)
                try:
                    audio = preprocess_audio(audio, audio_preprocessor, args.verbose, start)
                except Exception as exc:
                    return fail(
                        "Live audio preprocessing failed.",
                        [f"Runtime error: {exc}", NR_IMPORT_ERROR],
                        exit_code=6,
                    )
                if args.silence_detect and audio.size == 0:
                    continue

                try:
                    text = transcribe_chunk(pipe, audio, generate_kwargs)
                except Exception as exc:
                    return fail("Live transcription failed.", likely_reason_details(exc), exit_code=5)

                if text:
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


def parse_transcribe_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe with OpenVINO GenAI Whisper on NPU/GPU/CPU (file or live microphone mode)."
    )
    parser.add_argument("wav_path", type=pathlib.Path, nargs="?", help="Optional input .wav path.")
    parser.add_argument(
        "--device",
        default="NPU,GPU,CPU",
        help="Device preference order using NPU,GPU,CPU, or 'list' to print detected devices (default: NPU,GPU,CPU).",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help="Optional model directory. Defaults: NPU->whisper-base.en-int8-ov, GPU/CPU->whisper-tiny-fp16-ov.",
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
    enable_loopback_only_network()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv or raw_argv[0] not in {"transcribe", "web"}:
        raw_argv.insert(0, "transcribe")

    parser = argparse.ArgumentParser(description="Local Whisper transcription and WebRTC server.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("transcribe", help="Run file or live microphone transcription.")
    subparsers.add_parser("web", help="Run the browser audio streaming transcription server.")
    args = parser.parse_args(raw_argv[:1])

    if args.command == "web":
        import browser_webrtc

        return browser_webrtc.main(raw_argv[1:])
    return run_transcribe(raw_argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
