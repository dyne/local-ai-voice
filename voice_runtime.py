from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Callable

DEFAULT_DEVICE_ORDER = ("NPU", "GPU", "CPU")
DEFAULT_MODEL_FOR_DEVICE = {
    "NPU": "whisper-base.en-int8-ov",
    "GPU": "whisper-tiny-fp16-ov",
    "CPU": "whisper-tiny-fp16-ov",
}
DEFAULT_MODEL_REPO_FOR_DEVICE = {
    "NPU": "OpenVINO/whisper-base.en-int8-ov",
    "GPU": "OpenVINO/whisper-tiny-fp16-ov",
    "CPU": "OpenVINO/whisper-tiny-fp16-ov",
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
    if "Unsupported property STATIC_PIPELINE by CPU plugin" in message:
        details.append("This OpenVINO CPU plugin version does not support STATIC_PIPELINE.")
    details.append("Ensure OpenVINO runtime, drivers, and model export are compatible.")
    return details


def is_openvino_whisper_dir(model_dir: pathlib.Path) -> bool:
    return (model_dir / "openvino_encoder_model.xml").exists()


def _to_path(value: object) -> pathlib.Path:
    if isinstance(value, pathlib.Path):
        return value
    return pathlib.Path(str(value))


def _looks_like_hf_repo_id(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith("."):
        return False
    candidate = pathlib.Path(text)
    if candidate.is_absolute():
        return False
    return text.count("/") == 1 and "\\" not in text and " " not in text


def _download_openvino_model(repo_id: str) -> pathlib.Path:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise PipelineSetupError(
            "Model download is unavailable.",
            [f"Import error: {exc}", "Install huggingface_hub[hf_xet] and retry."],
        ) from exc

    # Reuse the shared Hugging Face cache first (no network).
    try:
        local_snapshot = snapshot_download(repo_id=repo_id, token=hf_token, local_files_only=True)
        local_model_dir = pathlib.Path(local_snapshot)
        if is_openvino_whisper_dir(local_model_dir):
            return local_model_dir
    except Exception:
        pass

    try:
        downloaded = snapshot_download(repo_id=repo_id, token=hf_token)
    except Exception as exc:
        raise PipelineSetupError(
            f"Failed to download model: {repo_id}",
            [f"Runtime error: {exc}", "Check network access, model repository name, and HF_TOKEN if authentication is required."],
        ) from exc
    model_dir = pathlib.Path(downloaded)
    if not is_openvino_whisper_dir(model_dir):
        raise PipelineSetupError(
            f"Downloaded model is not an OpenVINO Whisper export: {repo_id}",
            ["Expected openvino_encoder_model.xml in model directory."],
        )
    return model_dir


def resolve_model_dir(
    *,
    model_arg: object | None,
    selected_device: str,
    base_dir: pathlib.Path,
    offline: bool = False,
) -> pathlib.Path:
    default_repo = DEFAULT_MODEL_REPO_FOR_DEVICE[selected_device]
    if model_arg is None:
        bundled = base_dir / DEFAULT_MODEL_FOR_DEVICE[selected_device]
        if is_openvino_whisper_dir(bundled):
            return bundled
        if offline:
            raise PipelineSetupError(
                "Model download required but offline mode is enabled.",
                [
                    f"Selected device: {selected_device}",
                    f"Expected local model: {bundled}",
                    f"Disable --offline to download default model: {default_repo}",
                ],
            )
        return _download_openvino_model(default_repo)

    model_text = str(model_arg)
    model_path = _to_path(model_arg)
    if model_path.exists():
        if not is_openvino_whisper_dir(model_path):
            raise PipelineSetupError(
                f"Model directory is not an OpenVINO Whisper export: {model_path}",
                ["Expected openvino_encoder_model.xml in model directory."],
            )
        return model_path

    if _looks_like_hf_repo_id(model_text):
        if offline:
            raise PipelineSetupError(
                "Model download required but offline mode is enabled.",
                [
                    f"Requested repo: {model_text}",
                    "Disable --offline to download from Hugging Face.",
                ],
            )
        return _download_openvino_model(model_text)

    if offline:
        raise PipelineSetupError(
            "Model download required but offline mode is enabled.",
            [
                f"Missing local model path: {model_path}",
                f"Disable --offline to download default model: {default_repo}",
            ],
        )
    return _download_openvino_model(default_repo)


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

    model_dir = resolve_model_dir(
        model_arg=getattr(args, "model", None),
        selected_device=selected,
        base_dir=base_dir,
        offline=bool(getattr(args, "offline", False)),
    )
    try:
        import openvino_genai as ov_genai
    except Exception as exc:
        raise PipelineSetupError("openvino_genai is not available.", [f"Import error: {exc}", "Install OpenVINO GenAI and retry."]) from exc

    pipeline_kwargs: dict[str, object] = {"STATIC_PIPELINE": True}
    if selected == "CPU":
        # CPU plugin may reject STATIC_PIPELINE on some OpenVINO versions.
        try:
            pipe = ov_genai.WhisperPipeline(str(model_dir), selected, **pipeline_kwargs)
        except Exception:
            try:
                pipe = ov_genai.WhisperPipeline(str(model_dir), selected)
            except Exception as exc:
                raise PipelineSetupError(f"Failed to create WhisperPipeline on {selected}.", likely_reason_details(exc)) from exc
    else:
        try:
            pipe = ov_genai.WhisperPipeline(str(model_dir), selected, **pipeline_kwargs)
        except Exception as exc:
            raise PipelineSetupError(f"Failed to create WhisperPipeline on {selected}.", likely_reason_details(exc)) from exc

    if logger is not None:
        logger(f"Requested device order: {', '.join(preferred)}", verbose, start_time)
        logger("Backend: openvino", verbose, start_time)
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
