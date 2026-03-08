from __future__ import annotations

import argparse
import io
import pathlib

from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError
from local_ai.slices.voice.transcribe_runner import execute_transcribe_args


class FakeRuntime:
    def __init__(self) -> None:
        self.selected_device = "CPU"
        self.model_dir = pathlib.Path("model")
        self.pipe = object()
        self.generate_kwargs = {"task": "transcribe"}


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "wav_path": None,
        "verbose": True,
        "silence_detect": True,
        "vad_mode": 3,
        "vad_min_speech_frames": 2,
        "vad_min_speech_ratio": 0.3,
        "vad_min_utterance_ms": 200,
        "vad_hangover_ms": 150,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_execute_transcribe_args_prints_device_list_when_requested() -> None:
    stderr = io.StringIO()

    result = execute_transcribe_args(
        args=make_args(),
        perf_counter_fn=lambda: 1.0,
        configure_runtime_env_fn=lambda: None,
        create_runtime_fn=lambda **kwargs: (_ for _ in ()).throw(DeviceListRequested(["CPU", "GPU"])),
        create_audio_preprocessor_fn=lambda *args, **kwargs: object(),
        enable_loopback_only_network_fn=lambda: None,
        run_file_mode_fn=lambda *args: 1,
        run_live_mode_fn=lambda *args: 1,
        logger=lambda message, verbose, start_time: None,
        fail_fn=lambda reason, details=None, exit_code=1: exit_code,
        setup_error_exit_code_fn=lambda reason: 7,
        nr_import_error="nr missing",
        base_dir=pathlib.Path("."),
        stderr=stderr,
    )

    assert result == 0
    assert stderr.getvalue().splitlines() == ["CPU", "GPU"]


def test_execute_transcribe_args_returns_one_when_no_devices_exist() -> None:
    stderr = io.StringIO()

    result = execute_transcribe_args(
        args=make_args(),
        perf_counter_fn=lambda: 1.0,
        configure_runtime_env_fn=lambda: None,
        create_runtime_fn=lambda **kwargs: (_ for _ in ()).throw(DeviceListRequested([])),
        create_audio_preprocessor_fn=lambda *args, **kwargs: object(),
        enable_loopback_only_network_fn=lambda: None,
        run_file_mode_fn=lambda *args: 1,
        run_live_mode_fn=lambda *args: 1,
        logger=lambda message, verbose, start_time: None,
        fail_fn=lambda reason, details=None, exit_code=1: exit_code,
        setup_error_exit_code_fn=lambda reason: 7,
        nr_import_error="nr missing",
        base_dir=pathlib.Path("."),
        stderr=stderr,
    )

    assert result == 1
    assert "No OpenVINO devices detected" in stderr.getvalue()


def test_execute_transcribe_args_maps_pipeline_setup_errors() -> None:
    fail_calls: list[tuple[str, list[str], int]] = []

    result = execute_transcribe_args(
        args=make_args(),
        perf_counter_fn=lambda: 1.0,
        configure_runtime_env_fn=lambda: None,
        create_runtime_fn=lambda **kwargs: (_ for _ in ()).throw(PipelineSetupError("setup failed", ["detail"])),
        create_audio_preprocessor_fn=lambda *args, **kwargs: object(),
        enable_loopback_only_network_fn=lambda: None,
        run_file_mode_fn=lambda *args: 1,
        run_live_mode_fn=lambda *args: 1,
        logger=lambda message, verbose, start_time: None,
        fail_fn=lambda reason, details=None, exit_code=1: fail_calls.append((reason, details or [], exit_code)) or exit_code,
        setup_error_exit_code_fn=lambda reason: 5,
        nr_import_error="nr missing",
        base_dir=pathlib.Path("."),
        stderr=io.StringIO(),
    )

    assert result == 5
    assert fail_calls == [("setup failed", ["detail"], 5)]


def test_execute_transcribe_args_reports_audio_preprocessor_errors() -> None:
    fail_calls: list[tuple[str, list[str], int]] = []

    result = execute_transcribe_args(
        args=make_args(),
        perf_counter_fn=lambda: 1.0,
        configure_runtime_env_fn=lambda: None,
        create_runtime_fn=lambda **kwargs: FakeRuntime(),
        create_audio_preprocessor_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vad broken")),
        enable_loopback_only_network_fn=lambda: None,
        run_file_mode_fn=lambda *args: 1,
        run_live_mode_fn=lambda *args: 1,
        logger=lambda message, verbose, start_time: None,
        fail_fn=lambda reason, details=None, exit_code=1: fail_calls.append((reason, details or [], exit_code)) or exit_code,
        setup_error_exit_code_fn=lambda reason: 5,
        nr_import_error="nr missing",
        base_dir=pathlib.Path("."),
        stderr=io.StringIO(),
    )

    assert result == 6
    assert fail_calls == [("Audio preprocessing failed.", ["Runtime error: vad broken", "nr missing"], 6)]


def test_execute_transcribe_args_runs_file_mode_and_logs_completion() -> None:
    stderr = io.StringIO()
    perf_values = iter([1.0, 3.5])
    network_enabled: list[bool] = []
    logger_calls: list[tuple[str, bool, float]] = []
    file_calls: list[tuple[object, object, object, object, float]] = []

    result = execute_transcribe_args(
        args=make_args(wav_path=pathlib.Path("sample.wav")),
        perf_counter_fn=lambda: next(perf_values),
        configure_runtime_env_fn=lambda: None,
        create_runtime_fn=lambda **kwargs: FakeRuntime(),
        create_audio_preprocessor_fn=lambda *args, **kwargs: "preprocessor",
        enable_loopback_only_network_fn=lambda: network_enabled.append(True),
        run_file_mode_fn=lambda *args: file_calls.append(args) or 0,
        run_live_mode_fn=lambda *args: 1,
        logger=lambda message, verbose, start_time: logger_calls.append((message, verbose, start_time)),
        fail_fn=lambda reason, details=None, exit_code=1: exit_code,
        setup_error_exit_code_fn=lambda reason: 5,
        nr_import_error="nr missing",
        base_dir=pathlib.Path("."),
        stderr=stderr,
    )

    assert result == 0
    assert network_enabled == [True]
    assert len(file_calls) == 1
    assert logger_calls == [("Done in 2.50s", True, 1.0)]
    assert "Using device: CPU" in stderr.getvalue()
