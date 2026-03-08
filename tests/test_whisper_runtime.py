from __future__ import annotations

import pathlib
import sys
import types

import pytest

from local_ai.infrastructure.openvino import whisper
from local_ai.shared.domain.devices import DeviceSelection
from local_ai.shared.domain.errors import PipelineSetupError
from local_ai.shared.domain.models import ModelArtifact


def test_build_generate_kwargs_only_includes_requested_fields() -> None:
    args = types.SimpleNamespace(language="<|en|>", task="translate", timestamps=True)

    kwargs = whisper.build_generate_kwargs(args)

    assert kwargs == {"language": "<|en|>", "task": "translate", "return_timestamps": True}


def test_likely_reason_details_adds_known_hints() -> None:
    details = whisper.likely_reason_details(Exception("Unsupported property STATIC_PIPELINE by CPU plugin"))
    assert any("STATIC_PIPELINE" in line for line in details)


def test_create_whisper_runtime_uses_selected_device_and_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            calls.append((model_dir, device, kwargs))

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("GPU", "CPU"), available=["GPU.0"], selected="GPU"),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    logs: list[str] = []
    args = types.SimpleNamespace(device="GPU,CPU", model=None, offline=False, language=None, task=None, timestamps=False)

    runtime = whisper.create_whisper_runtime(
        args=args,
        base_dir=tmp_path,
        logger=lambda message, verbose, start: logs.append(message),
        verbose=True,
        start_time=1.0,
    )

    assert runtime.selected_device == "GPU"
    assert runtime.model_dir == tmp_path / "model"
    assert calls == [(str(tmp_path / "model"), "GPU", {"STATIC_PIPELINE": True})]
    assert any("Selected device: GPU" == line for line in logs)


def test_create_whisper_runtime_falls_back_for_cpu_without_static_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            calls.append((model_dir, device, kwargs))
            if kwargs.get("STATIC_PIPELINE") is True:
                raise RuntimeError("Unsupported property STATIC_PIPELINE by CPU plugin")

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("CPU",), available=["CPU"], selected="CPU"),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    args = types.SimpleNamespace(device="CPU", model=None, offline=False, language=None, task=None, timestamps=False)
    runtime = whisper.create_whisper_runtime(args=args, base_dir=tmp_path)

    assert runtime.selected_device == "CPU"
    assert calls == [
        (str(tmp_path / "model"), "CPU", {"STATIC_PIPELINE": True}),
        (str(tmp_path / "model"), "CPU", {}),
    ]


def test_create_whisper_runtime_raises_pipeline_setup_error_when_pipeline_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    class FakePipeline:
        def __init__(self, model_dir: str, device: str, **kwargs: object) -> None:
            raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(WhisperPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake_module)
    monkeypatch.setattr(
        whisper,
        "resolve_device_selection",
        lambda raw: DeviceSelection(requested=("GPU",), available=["GPU"], selected="GPU"),
    )
    monkeypatch.setattr(
        whisper,
        "resolve_model_artifact",
        lambda **kwargs: ModelArtifact(path=tmp_path / "model", source="local"),
    )

    args = types.SimpleNamespace(device="GPU", model=None, offline=False, language=None, task=None, timestamps=False)
    with pytest.raises(PipelineSetupError, match="Failed to create WhisperPipeline on GPU"):
        whisper.create_whisper_runtime(args=args, base_dir=tmp_path)
