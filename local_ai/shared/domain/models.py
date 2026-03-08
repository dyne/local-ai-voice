from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass

from local_ai.shared.domain.errors import PipelineSetupError

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


@dataclass(frozen=True)
class ModelArtifact:
    path: pathlib.Path
    source: str
    repo_id: str | None = None


def missing_openvino_whisper_files(model_dir: pathlib.Path) -> list[str]:
    return [name for name in REQUIRED_OPENVINO_WHISPER_FILES if not (model_dir / name).exists()]


def is_openvino_whisper_dir(model_dir: pathlib.Path) -> bool:
    return not missing_openvino_whisper_files(model_dir)


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


def huggingface_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def hf_cache_root() -> pathlib.Path:
    explicit = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (pathlib.Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None)
    )
    if explicit:
        return pathlib.Path(explicit)
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def resolve_cached_openvino_model(repo_id: str, hf_token: str | None) -> pathlib.Path | None:
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
    repo_cache_dir = hf_cache_root() / f"models--{org}--{name}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if is_openvino_whisper_dir(candidate):
            return candidate
    return None


def download_openvino_model(repo_id: str) -> ModelArtifact:
    hf_token = huggingface_token()
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise PipelineSetupError(
            "Model download is unavailable.",
            [f"Import error: {exc}", "Install huggingface_hub[hf_xet] and retry."],
        ) from exc

    cached_model_dir = resolve_cached_openvino_model(repo_id, hf_token)
    if cached_model_dir is not None:
        return ModelArtifact(path=cached_model_dir, source="hf-cache", repo_id=repo_id)

    try:
        downloaded = pathlib.Path(snapshot_download(repo_id=repo_id, token=hf_token))
    except Exception as exc:
        raise PipelineSetupError(
            f"Failed to download model: {repo_id}",
            [f"Runtime error: {exc}", "Check network access, model repository name, and HF_TOKEN if authentication is required."],
        ) from exc

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        cached_model_dir = resolve_cached_openvino_model(repo_id, hf_token)
        if cached_model_dir is not None:
            return ModelArtifact(path=cached_model_dir, source="hf-cache", repo_id=repo_id)
        if is_openvino_whisper_dir(downloaded):
            return ModelArtifact(path=downloaded, source="download", repo_id=repo_id)
        time.sleep(0.1)

    missing_files = missing_openvino_whisper_files(downloaded)
    raise PipelineSetupError(
        f"Downloaded model is not an OpenVINO Whisper export: {repo_id}",
        [
            f"Downloaded path: {downloaded}",
            "Missing required files: " + ", ".join(missing_files),
        ],
    )


def resolve_model_artifact(
    *,
    model_arg: object | None,
    selected_device: str,
    base_dir: pathlib.Path,
    offline: bool = False,
) -> ModelArtifact:
    default_repo = DEFAULT_MODEL_REPO_FOR_DEVICE[selected_device]
    hf_token = huggingface_token()
    if model_arg is None:
        bundled = base_dir / DEFAULT_MODEL_FOR_DEVICE[selected_device]
        if is_openvino_whisper_dir(bundled):
            return ModelArtifact(path=bundled, source="bundled", repo_id=default_repo)
        cached = resolve_cached_openvino_model(default_repo, hf_token)
        if cached is not None:
            return ModelArtifact(path=cached, source="hf-cache", repo_id=default_repo)
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
        return download_openvino_model(default_repo)

    model_text = str(model_arg)
    model_path = _to_path(model_arg)
    if model_path.exists():
        if not is_openvino_whisper_dir(model_path):
            missing_files = missing_openvino_whisper_files(model_path)
            raise PipelineSetupError(
                f"Model directory is not an OpenVINO Whisper export: {model_path}",
                ["Missing required files: " + ", ".join(missing_files)],
            )
        return ModelArtifact(path=model_path, source="local")

    repo_id = _parse_hf_repo_id(model_text)
    if repo_id is not None:
        cached = resolve_cached_openvino_model(repo_id, hf_token)
        if cached is not None:
            return ModelArtifact(path=cached, source="hf-cache", repo_id=repo_id)
        if offline:
            raise PipelineSetupError(
                "Model download required but offline mode is enabled.",
                [
                    f"Requested repo: {repo_id}",
                    "The requested repo was not found in the shared Hugging Face cache.",
                    "Disable --offline to download from Hugging Face.",
                ],
            )
        return download_openvino_model(repo_id)

    raise PipelineSetupError(
        f"Model directory not found: {model_path}",
        [
            "If this is a local path, verify it exists and contains openvino_encoder_model.xml.",
            "If this is a Hugging Face repo id, use the form org/name (for example OpenVINO/whisper-tiny-fp16-ov).",
        ],
    )


def resolve_model_dir(
    *,
    model_arg: object | None,
    selected_device: str,
    base_dir: pathlib.Path,
    offline: bool = False,
) -> pathlib.Path:
    return resolve_model_artifact(
        model_arg=model_arg,
        selected_device=selected_device,
        base_dir=base_dir,
        offline=offline,
    ).path
