from __future__ import annotations

import pytest

from local_ai.shared.domain import devices
from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError


def test_parse_device_preference_keeps_order() -> None:
    assert devices.parse_device_preference("gpu,cpu") == ("GPU", "CPU")


def test_parse_device_preference_list_keyword() -> None:
    assert devices.parse_device_preference("list") == ("LIST",)


def test_parse_device_preference_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        devices.parse_device_preference("GPU,TPU")


def test_pick_first_available_device_matches_prefixed_devices() -> None:
    selected = devices.pick_first_available_device(("NPU", "GPU"), ["GPU.0", "CPU"])
    assert selected == "GPU"


def test_resolve_device_selection_returns_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "query_available_devices", lambda: ["GPU.0", "CPU"])

    selection = devices.resolve_device_selection("NPU,GPU,CPU")

    assert selection.requested == ("NPU", "GPU", "CPU")
    assert selection.available == ["GPU.0", "CPU"]
    assert selection.selected == "GPU"


def test_resolve_device_selection_raises_for_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "query_available_devices", lambda: ["CPU"])

    with pytest.raises(DeviceListRequested) as exc:
        devices.resolve_device_selection("list")

    assert exc.value.devices == ["CPU"]


def test_resolve_device_selection_raises_when_no_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "query_available_devices", lambda: [])

    with pytest.raises(PipelineSetupError, match="No OpenVINO devices detected"):
        devices.resolve_device_selection("CPU")


def test_resolve_device_selection_raises_when_requested_devices_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices, "query_available_devices", lambda: ["CPU"])

    with pytest.raises(PipelineSetupError, match="No requested device is available"):
        devices.resolve_device_selection("NPU,GPU")
