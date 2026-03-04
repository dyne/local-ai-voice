#!/usr/bin/env python
import argparse
import os
import pathlib
import sys
import time
import wave

import numpy as np

TARGET_SAMPLE_RATE = 16000
DEFAULT_DEVICE_ORDER = ("NPU", "GPU", "CPU")
DEFAULT_MODEL_FOR_DEVICE = {
    "NPU": "whisper-base.en-int8-ov",
    "GPU": "whisper-tiny-fp16-ov",
    "CPU": "whisper-tiny-fp16-ov",
}
SILENCE_DB_THRESHOLD = -45.0
SILENCE_MIN_LENGTH_MS = 5000
SILENCE_MIN_INTERVAL_MS = 300
SILENCE_HOP_SIZE_MS = 10
SILENCE_MAX_KEPT_MS = 700
LIVE_SILENCE_MIN_LENGTH_MS = 300
LIVE_SILENCE_MIN_INTERVAL_MS = 120
LIVE_SILENCE_MAX_KEPT_MS = 220


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


def discard_silence(
    audio: np.ndarray,
    sample_rate: int,
    verbose: bool,
    start: float,
    *,
    min_length_ms: int = SILENCE_MIN_LENGTH_MS,
    min_interval_ms: int = SILENCE_MIN_INTERVAL_MS,
    max_sil_kept_ms: int = SILENCE_MAX_KEPT_MS,
) -> np.ndarray:
    try:
        from slicer import Slicer
    except Exception as exc:
        raise RuntimeError("Failed to import slicer.py for silence detection.") from exc

    slicer = Slicer(
        sr=sample_rate,
        threshold=SILENCE_DB_THRESHOLD,
        min_length=min_length_ms,
        min_interval=min_interval_ms,
        hop_size=SILENCE_HOP_SIZE_MS,
        max_sil_kept=max_sil_kept_ms,
    )
    chunks = slicer.slice(audio)
    non_empty = [np.asarray(chunk, dtype=np.float32) for chunk in chunks if np.asarray(chunk).size > 0]
    if not non_empty:
        return np.asarray([], dtype=np.float32)
    if len(non_empty) == 1:
        trimmed = non_empty[0]
    else:
        trimmed = np.concatenate(non_empty).astype(np.float32, copy=False)

    removed = max(0, int(audio.shape[0] - trimmed.shape[0]))
    if removed > 0:
        log(
            f"Silence trimmed: removed {removed / float(sample_rate):.2f}s, kept {trimmed.shape[0] / float(sample_rate):.2f}s",
            verbose,
            start,
        )
    return trimmed


def parse_device_preference(raw: str) -> tuple[str, ...]:
    stripped = raw.strip().upper()
    if stripped == "LIST":
        return ("LIST",)
    parts = [p.strip() for p in stripped.split(",") if p.strip()]
    if not parts:
        raise ValueError("Device list is empty.")
    invalid = [p for p in parts if p not in DEFAULT_DEVICE_ORDER]
    if invalid:
        raise ValueError(f"Unsupported device value(s): {', '.join(invalid)}. Use only NPU,GPU,CPU or list.")
    return tuple(parts)


def query_available_devices() -> list[str]:
    try:
        import openvino as ov

        return list(ov.Core().available_devices)
    except Exception:
        try:
            from openvino.runtime import Core

            return list(Core().available_devices)
        except Exception:
            return []


def configure_openvino_runtime_env() -> None:
    # Frozen builds may need explicit plugin search paths for OpenVINO runtime.
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


def pick_first_available_device(preferred: tuple[str, ...], available: list[str]) -> str | None:
    for want in preferred:
        if any(dev == want or dev.startswith(f"{want}.") for dev in available):
            return want
    return None


def resolve_model_dir(args: argparse.Namespace, selected_device: str) -> pathlib.Path:
    if args.model is not None:
        return args.model
    return pathlib.Path(__file__).resolve().parent / DEFAULT_MODEL_FOR_DEVICE[selected_device]


def build_generate_kwargs(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if args.language:
        kwargs["language"] = args.language
    if args.task:
        kwargs["task"] = args.task
    if args.timestamps:
        kwargs["return_timestamps"] = True
    return kwargs


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


def likely_reason_details(exc: Exception) -> list[str]:
    message = str(exc)
    details = [f"Runtime error: {message}"]
    if "Upper bounds were not specified" in message:
        details.append("NPU compiler rejected dynamic bounds for this model/runtime combination.")
    if "OpenVINO and OpenVINO Tokenizers versions are not binary compatible" in message:
        details.append("Align package versions: pip install -U openvino openvino-genai openvino-tokenizers")
    if "openvino_tokenizers.dll" in message:
        details.append("Frozen build is missing tokenizer runtime library; rebuild with tokenizer binaries/data included.")
    details.append("Ensure OpenVINO runtime, drivers, and model export are compatible.")
    return details


def transcribe_chunk(pipe: object, audio: np.ndarray, generate_kwargs: dict[str, object]) -> str:
    result = pipe.generate(audio.tolist(), **generate_kwargs)
    return result_to_text(result).strip()


def run_file_mode(
    args: argparse.Namespace,
    pipe: object,
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
        audio = discard_silence(audio, sample_rate, args.verbose, start)
    except Exception as exc:
        return fail(
            "Silence detection failed.",
            [f"Runtime error: {exc}", "Ensure slicer.py is present and compatible."],
            exit_code=6,
        )
    if audio.size == 0:
        return fail("No non-silent audio detected.", ["Input appears silent after trimming."], exit_code=2)
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
                original_audio = audio
                try:
                    detected_audio = discard_silence(
                        audio,
                        TARGET_SAMPLE_RATE,
                        args.verbose,
                        start,
                        min_length_ms=LIVE_SILENCE_MIN_LENGTH_MS,
                        min_interval_ms=LIVE_SILENCE_MIN_INTERVAL_MS,
                        max_sil_kept_ms=LIVE_SILENCE_MAX_KEPT_MS,
                    )
                except Exception as exc:
                    return fail(
                        "Live silence detection failed.",
                        [f"Runtime error: {exc}", "Ensure slicer.py is present and compatible."],
                        exit_code=6,
                    )
                if detected_audio.size == 0:
                    continue
                # In live mode, keep original chunk once speech is detected to avoid clipping word onsets.
                audio = original_audio

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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Live mode chunk duration in seconds (used when wav_path is omitted).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    configure_openvino_runtime_env()

    try:
        preferred_devices = parse_device_preference(args.device)
    except ValueError as exc:
        return fail(str(exc), exit_code=2)

    available = query_available_devices()
    if preferred_devices == ("LIST",):
        if not available:
            print("No OpenVINO devices detected")
            return 1
        for dev in available:
            print(dev)
        return 0
    if not available:
        details = ["Check OpenVINO installation and drivers."]
        if getattr(sys, "frozen", False):
            details.append("Frozen build may be missing OpenVINO runtime plugins (CPU/NPU/GPU). Rebuild with collected binaries/data.")
        return fail("No OpenVINO devices detected.", details, exit_code=3)

    selected_device = pick_first_available_device(preferred_devices, available)
    if selected_device is None:
        return fail(
            "No requested device is available.",
            [f"Requested order: {', '.join(preferred_devices)}", f"Detected devices: {', '.join(available)}"],
            exit_code=3,
        )
    log(f"Requested device order: {', '.join(preferred_devices)}", args.verbose, start)
    log(f"Selected device: {selected_device}", args.verbose, start)

    model_dir = resolve_model_dir(args, selected_device)
    if not model_dir.exists():
        return fail(
            f"Model directory not found: {model_dir}",
            [f"Selected device: {selected_device}", "Pass --model with a valid local Whisper OpenVINO model directory."],
            exit_code=2,
        )
    log(f"Model directory: {model_dir}", args.verbose, start)

    try:
        import openvino_genai as ov_genai
    except Exception as exc:
        return fail(
            "openvino_genai is not available.",
            [f"Import error: {exc}", "Install OpenVINO GenAI and retry."],
            exit_code=3,
        )

    try:
        # STATIC_PIPELINE is required for stable Whisper execution on NPU.
        pipe = ov_genai.WhisperPipeline(str(model_dir), selected_device, STATIC_PIPELINE=True)
    except Exception as exc:
        return fail(
            f"Failed to create WhisperPipeline on {selected_device}.",
            likely_reason_details(exc),
            exit_code=4,
        )

    generate_kwargs = build_generate_kwargs(args)
    status = run_file_mode(args, pipe, generate_kwargs, start) if args.wav_path else run_live_mode(args, pipe, generate_kwargs, start)
    if status == 0:
        log(f"Done in {time.perf_counter() - start:.2f}s", args.verbose, start)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
