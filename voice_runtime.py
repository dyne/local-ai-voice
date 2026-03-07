from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Callable

DEFAULT_DEVICE_ORDER = ("NPU", "GPU", "CPU")
DEFAULT_MODEL_FOR_DEVICE = {
    "NPU": "whisper-base.en-int8-ov",
    "GPU": "whisper-tiny-fp16-ov",
    "CPU": "whisper-tiny-fp16-ov",
}


@dataclass
class PipelineSetupError(Exception):
    reason: str
    details: list[str]


@dataclass
class DeviceListRequested(Exception):
    devices: list[str]


@dataclass
class WhisperRuntime:
    pipe: object
    generate_kwargs: dict[str, object]
    selected_device: str
    model_dir: pathlib.Path
    preferred_devices: tuple[str, ...]
    available_devices: list[str]


def build_generate_kwargs(args: object) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    language = getattr(args, "language", None)
    task = getattr(args, "task", None)
    timestamps = bool(getattr(args, "timestamps", False))
    if language:
        kwargs["language"] = language
    if task:
        kwargs["task"] = task
    if timestamps:
        kwargs["return_timestamps"] = True
    return kwargs


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


def pick_first_available_device(preferred: tuple[str, ...], available: list[str]) -> str | None:
    for want in preferred:
        if any(dev == want or dev.startswith(f"{want}.") for dev in available):
            return want
    return None


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


def create_whisper_runtime(
    *,
    args: object,
    base_dir: pathlib.Path,
    logger: Callable[[str, bool, float | None], None] | None = None,
    verbose: bool = False,
    start_time: float | None = None,
) -> WhisperRuntime:
    preferred = parse_device_preference(getattr(args, "device"))
    available = query_available_devices()

    if preferred == ("LIST",):
        raise DeviceListRequested(devices=available)
    if not available:
        raise PipelineSetupError("No OpenVINO devices detected.", ["Check OpenVINO installation and drivers."])

    selected = pick_first_available_device(preferred, available)
    if selected is None:
        raise PipelineSetupError(
            "No requested device is available.",
            [f"Requested order: {', '.join(preferred)}", f"Detected devices: {', '.join(available)}"],
        )

    model_arg = getattr(args, "model", None)
    model_dir = model_arg if model_arg is not None else base_dir / DEFAULT_MODEL_FOR_DEVICE[selected]
    if not model_dir.exists():
        raise PipelineSetupError(
            f"Model directory not found: {model_dir}",
            [f"Selected device: {selected}", "Pass --model with a valid local Whisper OpenVINO model directory."],
        )

    try:
        import openvino_genai as ov_genai
    except Exception as exc:
        raise PipelineSetupError("openvino_genai is not available.", [f"Import error: {exc}", "Install OpenVINO GenAI and retry."])

    try:
        pipe = ov_genai.WhisperPipeline(str(model_dir), selected, STATIC_PIPELINE=True)
    except Exception as exc:
        raise PipelineSetupError(f"Failed to create WhisperPipeline on {selected}.", likely_reason_details(exc))

    if logger is not None:
        logger(f"Requested device order: {', '.join(preferred)}", verbose, start_time)
        logger(f"Selected device: {selected}", verbose, start_time)
        logger(f"Model directory: {model_dir}", verbose, start_time)

    return WhisperRuntime(
        pipe=pipe,
        generate_kwargs=build_generate_kwargs(args),
        selected_device=selected,
        model_dir=model_dir,
        preferred_devices=preferred,
        available_devices=available,
    )
