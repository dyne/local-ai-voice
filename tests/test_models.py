from __future__ import annotations

import pathlib

import pytest

from local_ai.shared.domain.errors import PipelineSetupError
from local_ai.shared.domain.models import (
    REQUIRED_OPENVINO_WHISPER_FILES,
    ModelArtifact,
    is_openvino_whisper_dir,
    missing_openvino_whisper_files,
    resolve_model_artifact,
)


def create_openvino_model_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_OPENVINO_WHISPER_FILES:
        (path / name).write_text("x", encoding="utf-8")
    return path


def test_is_openvino_whisper_dir_requires_all_files(tmp_path: pathlib.Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / REQUIRED_OPENVINO_WHISPER_FILES[0]).write_text("x", encoding="utf-8")

    assert not is_openvino_whisper_dir(model_dir)
    assert len(missing_openvino_whisper_files(model_dir)) == len(REQUIRED_OPENVINO_WHISPER_FILES) - 1


def test_resolve_model_artifact_prefers_bundled_default(tmp_path: pathlib.Path) -> None:
    bundled = create_openvino_model_dir(tmp_path / "whisper-tiny-fp16-ov")

    artifact = resolve_model_artifact(
        model_arg=None,
        selected_device="CPU",
        base_dir=tmp_path,
        offline=True,
    )

    assert artifact == ModelArtifact(path=bundled, source="bundled", repo_id="OpenVINO/whisper-tiny-fp16-ov")


def test_resolve_model_artifact_accepts_explicit_local_dir(tmp_path: pathlib.Path) -> None:
    model_dir = create_openvino_model_dir(tmp_path / "custom-model")

    artifact = resolve_model_artifact(
        model_arg=model_dir,
        selected_device="CPU",
        base_dir=tmp_path,
        offline=False,
    )

    assert artifact == ModelArtifact(path=model_dir, source="local")


def test_resolve_model_artifact_rejects_incomplete_local_dir(tmp_path: pathlib.Path) -> None:
    model_dir = tmp_path / "broken-model"
    model_dir.mkdir()

    with pytest.raises(PipelineSetupError, match="Model directory is not an OpenVINO Whisper export"):
        resolve_model_artifact(
            model_arg=model_dir,
            selected_device="CPU",
            base_dir=tmp_path,
            offline=False,
        )


def test_resolve_model_artifact_uses_cached_default_before_offline_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = create_openvino_model_dir(tmp_path / "cache" / "snap")
    monkeypatch.setattr("local_ai.shared.domain.models.resolve_cached_openvino_model", lambda repo_id, hf_token: cached)

    artifact = resolve_model_artifact(
        model_arg=None,
        selected_device="CPU",
        base_dir=tmp_path,
        offline=True,
    )

    assert artifact == ModelArtifact(path=cached, source="hf-cache", repo_id="OpenVINO/whisper-tiny-fp16-ov")


def test_resolve_model_artifact_fails_offline_without_cached_default(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_ai.shared.domain.models.resolve_cached_openvino_model", lambda repo_id, hf_token: None)

    with pytest.raises(PipelineSetupError, match="offline mode is enabled"):
        resolve_model_artifact(
            model_arg=None,
            selected_device="CPU",
            base_dir=tmp_path,
            offline=True,
        )


def test_resolve_model_artifact_uses_cached_repo_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = create_openvino_model_dir(tmp_path / "cache" / "repo")
    monkeypatch.setattr("local_ai.shared.domain.models.resolve_cached_openvino_model", lambda repo_id, hf_token: cached)

    artifact = resolve_model_artifact(
        model_arg="OpenVINO/whisper-tiny-fp16-ov",
        selected_device="CPU",
        base_dir=tmp_path,
        offline=True,
    )

    assert artifact == ModelArtifact(path=cached, source="hf-cache", repo_id="OpenVINO/whisper-tiny-fp16-ov")


def test_resolve_model_artifact_downloads_repo_id_when_allowed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = create_openvino_model_dir(tmp_path / "downloaded")
    monkeypatch.setattr("local_ai.shared.domain.models.resolve_cached_openvino_model", lambda repo_id, hf_token: None)
    monkeypatch.setattr(
        "local_ai.shared.domain.models.download_openvino_model",
        lambda repo_id: ModelArtifact(path=downloaded, source="download", repo_id=repo_id),
    )

    artifact = resolve_model_artifact(
        model_arg="OpenVINO/whisper-tiny-fp16-ov",
        selected_device="CPU",
        base_dir=tmp_path,
        offline=False,
    )

    assert artifact == ModelArtifact(path=downloaded, source="download", repo_id="OpenVINO/whisper-tiny-fp16-ov")


def test_resolve_model_artifact_fails_for_missing_repo_id_in_offline_mode(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("local_ai.shared.domain.models.resolve_cached_openvino_model", lambda repo_id, hf_token: None)

    with pytest.raises(PipelineSetupError, match="Requested repo: OpenVINO/whisper-tiny-fp16-ov"):
        resolve_model_artifact(
            model_arg="OpenVINO/whisper-tiny-fp16-ov",
            selected_device="CPU",
            base_dir=tmp_path,
            offline=True,
        )


def test_resolve_model_artifact_fails_for_missing_local_path(tmp_path: pathlib.Path) -> None:
    with pytest.raises(PipelineSetupError, match="Model directory not found"):
        resolve_model_artifact(
            model_arg=tmp_path / "missing-model",
            selected_device="CPU",
            base_dir=tmp_path,
            offline=False,
        )
