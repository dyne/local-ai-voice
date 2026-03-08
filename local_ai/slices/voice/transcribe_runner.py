from __future__ import annotations

import pathlib
from typing import TextIO

from local_ai.shared.domain.errors import DeviceListRequested, PipelineSetupError


def execute_transcribe_args(
    *,
    args: object,
    perf_counter_fn,
    configure_runtime_env_fn,
    create_runtime_fn,
    create_audio_preprocessor_fn,
    enable_loopback_only_network_fn,
    run_file_mode_fn,
    run_live_mode_fn,
    logger,
    fail_fn,
    setup_error_exit_code_fn,
    nr_import_error: str,
    base_dir: pathlib.Path,
    stderr: TextIO,
) -> int:
    start = perf_counter_fn()
    configure_runtime_env_fn()

    try:
        runtime = create_runtime_fn(
            args=args,
            base_dir=base_dir,
            logger=logger,
            verbose=getattr(args, "verbose"),
            start_time=start,
        )
    except DeviceListRequested as exc:
        if not exc.devices:
            print("No OpenVINO devices detected", file=stderr)
            return 1
        for device in exc.devices:
            print(device, file=stderr)
        return 0
    except PipelineSetupError as exc:
        return fail_fn(exc.reason, exc.details, exit_code=setup_error_exit_code_fn(exc.reason))

    print(f"Using device: {runtime.selected_device}", file=stderr, flush=True)
    print(f"Using model: {runtime.model_dir}", file=stderr, flush=True)
    enable_loopback_only_network_fn()

    try:
        audio_preprocessor = create_audio_preprocessor_fn(
            getattr(args, "silence_detect"),
            vad_mode=getattr(args, "vad_mode"),
            min_speech_frames=getattr(args, "vad_min_speech_frames"),
            min_speech_ratio=getattr(args, "vad_min_speech_ratio"),
            min_utterance_ms=getattr(args, "vad_min_utterance_ms"),
            hangover_ms=getattr(args, "vad_hangover_ms"),
        )
    except Exception as exc:
        return fail_fn("Audio preprocessing failed.", [f"Runtime error: {exc}", nr_import_error], exit_code=6)

    status = (
        run_file_mode_fn(args, runtime.pipe, audio_preprocessor, runtime.generate_kwargs, start)
        if getattr(args, "wav_path")
        else run_live_mode_fn(args, runtime.pipe, audio_preprocessor, runtime.generate_kwargs, start)
    )
    if status == 0:
        logger(f"Done in {perf_counter_fn() - start:.2f}s", getattr(args, "verbose"), start)
    return status
