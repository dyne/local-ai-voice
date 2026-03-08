from __future__ import annotations

import pathlib
from typing import Callable

from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError


def create_server_context(
    *,
    args: object,
    start_time: float,
    configure_runtime_env: Callable[[], None],
    create_runtime: Callable[..., object],
    logger: Callable[[str, bool, float | None], None],
    base_dir: pathlib.Path,
    context_factory: Callable[..., object],
    lock_factory: Callable[[], object],
) -> object:
    configure_runtime_env()
    try:
        runtime = create_runtime(
            args=args,
            base_dir=base_dir,
            logger=logger,
            verbose=getattr(args, "verbose"),
            start_time=start_time,
        )
    except DeviceListRequested as exc:
        if not exc.devices:
            raise RuntimeError("No OpenVINO devices detected.") from exc
        raise RuntimeError("Devices: " + ", ".join(exc.devices)) from exc
    except PipelineSetupError as exc:
        raise RuntimeError(f"{exc.reason} {' '.join(exc.details)}") from exc

    return context_factory(
        pipe=runtime.pipe,
        generate_kwargs=runtime.generate_kwargs,
        selected_device=runtime.selected_device,
        model_dir=runtime.model_dir,
        silence_detect_default=getattr(args, "silence_detect"),
        vad_mode_default=getattr(args, "vad_mode"),
        chunk_seconds=getattr(args, "chunk_seconds"),
        overlap_seconds=getattr(args, "overlap_seconds"),
        verbose=getattr(args, "verbose"),
        start_time=start_time,
        infer_lock=lock_factory(),
    )
