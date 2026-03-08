from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineSetupError(Exception):
    reason: str
    details: list[str]


@dataclass
class DeviceListRequested(Exception):
    devices: list[str]
