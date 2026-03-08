from __future__ import annotations

import os
import pathlib
import time
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
REQUIRED_OPENVINO_WHISPER_FILES = (
    "openvino_encoder_model.xml",
    "openvino_encoder_model.bin",
    "openvino_decoder_model.xml",
    "openvino_decoder_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    "config.json",
    "generation_config.json",
)


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
    return not missing_openvino_whisper_files(model_dir)


def missing_openvino_whisper_files(model_dir: pathlib.Path) -> list[str]:
    return [name for name in REQUIRED_OPENVINO_WHISPER_FILES if not (model_dir / name).exists()]


def _to_path(value: object) -> pathlib.Path:
    if isinstance(value, pathlib.Path):
        return value
    return pathlib.Path(str(value))


def _parse_hf_repo_id(value: str) -> str | None:
    text = value.strip()
    if not text or text.startswith("."):
        return None
    candidate: pathlib.PurePath
    if "\\" in text:
        candidate = pathlib.PureWindowsPath(text)
    else:
        candidate = pathlib.PurePosixPath(text)
    if candidate.is_absolute():
        return None
    parts = [part for part in candidate.parts if part not in {"", "."}]
    if len(parts) != 2:
        return None
    if any(part in {".."} or " " in part for part in parts):
        return None
    return f"{parts[0]}/{parts[1]}"


def _hf_cache_root() -> pathlib.Path:
    explicit = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (pathlib.Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None)
    )
    if explicit:
        return pathlib.Path(explicit)
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_cached_openvino_model(repo_id: str, hf_token: str | None) -> pathlib.Path | None:
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        snapshot_download = None

    if snapshot_download is not None:
        try:
            local_snapshot = pathlib.Path(snapshot_download(repo_id=repo_id, token=hf_token, local_files_only=True))
        except Exception:
            local_snapshot = None
        if local_snapshot is not None and is_openvino_whisper_dir(local_snapshot):
            return local_snapshot

    org, name = repo_id.split("/", 1)
    repo_cache_dir = _hf_cache_root() / f"models--{org}--{name}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if is_openvino_whisper_dir(candidate):
            return candidate
    return None


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
    cached_model_dir = _resolve_cached_openvino_model(repo_id, hf_token)
    if cached_model_dir is not None:
        return cached_model_dir

    try:
        downloaded = pathlib.Path(snapshot_download(repo_id=repo_id, token=hf_token))
    except Exception as exc:
        raise PipelineSetupError(
            f"Failed to download model: {repo_id}",
            [f"Runtime error: {exc}", "Check network access, model repository name, and HF_TOKEN if authentication is required."],
        ) from exc

    # In frozen builds, the freshly downloaded snapshot can be visible before all files are
    # immediately observable at the returned path. Re-resolve through the cache and wait briefly.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        cached_model_dir = _resolve_cached_openvino_model(repo_id, hf_token)
        if cached_model_dir is not None:
            return cached_model_dir
        if is_openvino_whisper_dir(downloaded):
            return downloaded
        time.sleep(0.1)

    missing_files = missing_openvino_whisper_files(downloaded)
    if missing_files:
        raise PipelineSetupError(
            f"Downloaded model is not an OpenVINO Whisper export: {repo_id}",
            [
                f"Downloaded path: {downloaded}",
                "Missing required files: " + ", ".join(missing_files),
            ],
        )
    return downloaded


def resolve_model_dir(
    *,
    model_arg: object | None,
    selected_device: str,
    base_dir: pathlib.Path,
    offline: bool = False,
) -> pathlib.Path:
    default_repo = DEFAULT_MODEL_REPO_FOR_DEVICE[selected_device]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if model_arg is None:
        bundled = base_dir / DEFAULT_MODEL_FOR_DEVICE[selected_device]
        if is_openvino_whisper_dir(bundled):
            return bundled
        cached = _resolve_cached_openvino_model(default_repo, hf_token)
        if cached is not None:
            return cached
        if offline:
            raise PipelineSetupError(
                "Model download required but offline mode is enabled.",
                [
                    f"Selected device: {selected_device}",
                    f"Expected local model: {bundled}",
                    f"Checked Hugging Face cache for: {default_repo}",
                    f"Disable --offline to download default model: {default_repo}",
                ],
            )
        return _download_openvino_model(default_repo)

    model_text = str(model_arg)
    model_path = _to_path(model_arg)
    if model_path.exists():
        if not is_openvino_whisper_dir(model_path):
            missing_files = missing_openvino_whisper_files(model_path)
            raise PipelineSetupError(
                f"Model directory is not an OpenVINO Whisper export: {model_path}",
                ["Missing required files: " + ", ".join(missing_files)],
            )
        return model_path

    repo_id = _parse_hf_repo_id(model_text)
    if repo_id is not None:
        cached = _resolve_cached_openvino_model(repo_id, hf_token)
        if cached is not None:
            return cached
        if offline:
            raise PipelineSetupError(
                "Model download required but offline mode is enabled.",
                [
                    f"Requested repo: {repo_id}",
                    "The requested repo was not found in the shared Hugging Face cache.",
                    "Disable --offline to download from Hugging Face.",
                ],
            )
        return _download_openvino_model(repo_id)

    raise PipelineSetupError(
        f"Model directory not found: {model_path}",
        [
            "If this is a local path, verify it exists and contains openvino_encoder_model.xml.",
            "If this is a Hugging Face repo id, use the form org/name (for example OpenVINO/whisper-tiny-fp16-ov).",
        ],
    )


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
