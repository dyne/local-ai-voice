from __future__ import annotations

from dataclasses import dataclass

from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError

DEFAULT_DEVICE_ORDER = ("NPU", "GPU", "CPU")


@dataclass(frozen=True)
class DeviceSelection:
    requested: tuple[str, ...]
    available: list[str]
    selected: str


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


def resolve_device_selection(raw_preference: str) -> DeviceSelection:
    preferred = parse_device_preference(raw_preference)
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

    return DeviceSelection(requested=preferred, available=available, selected=selected)
