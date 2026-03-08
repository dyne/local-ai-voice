from __future__ import annotations

import argparse
import asyncio
import pathlib
import types

import pytest

from browser_webrtc import ServerContext
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError
from local_ai.slices.voice.web_ui.runtime_context import create_server_context


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        device="CPU",
        model=None,
        offline=False,
        language=None,
        task=None,
        timestamps=False,
        silence_detect=True,
        vad_mode=3,
        chunk_seconds=1.5,
        overlap_seconds=0.0,
        verbose=False,
    )


def test_create_server_context_returns_server_context() -> None:
    runtime = types.SimpleNamespace(
        pipe=object(),
        generate_kwargs={"language": "<|en|>"},
        selected_device="CPU",
        model_dir=pathlib.Path("model"),
    )

    ctx = create_server_context(
        args=make_args(),
        start_time=1.0,
        configure_runtime_env=lambda: None,
        create_runtime=lambda **kwargs: runtime,
        logger=lambda message, verbose, start: None,
        base_dir=pathlib.Path("."),
        context_factory=ServerContext,
        lock_factory=asyncio.Lock,
    )

    assert ctx.selected_device == "CPU"
    assert ctx.model_dir == pathlib.Path("model")
    assert ctx.generate_kwargs == {"language": "<|en|>"}


def test_create_server_context_maps_empty_device_list_to_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="No OpenVINO devices detected"):
        create_server_context(
            args=make_args(),
            start_time=1.0,
            configure_runtime_env=lambda: None,
            create_runtime=lambda **kwargs: (_ for _ in ()).throw(DeviceListRequested(devices=[])),
            logger=lambda message, verbose, start: None,
            base_dir=pathlib.Path("."),
            context_factory=ServerContext,
            lock_factory=asyncio.Lock,
        )


def test_create_server_context_maps_device_list_to_runtime_error_message() -> None:
    with pytest.raises(RuntimeError, match="Devices: GPU, CPU"):
        create_server_context(
            args=make_args(),
            start_time=1.0,
            configure_runtime_env=lambda: None,
            create_runtime=lambda **kwargs: (_ for _ in ()).throw(DeviceListRequested(devices=["GPU", "CPU"])),
            logger=lambda message, verbose, start: None,
            base_dir=pathlib.Path("."),
            context_factory=ServerContext,
            lock_factory=asyncio.Lock,
        )


def test_create_server_context_maps_pipeline_setup_error() -> None:
    with pytest.raises(RuntimeError, match="bad reason detail one detail two"):
        create_server_context(
            args=make_args(),
            start_time=1.0,
            configure_runtime_env=lambda: None,
            create_runtime=lambda **kwargs: (_ for _ in ()).throw(PipelineSetupError("bad reason", ["detail one", "detail two"])),
            logger=lambda message, verbose, start: None,
            base_dir=pathlib.Path("."),
            context_factory=ServerContext,
            lock_factory=asyncio.Lock,
        )
