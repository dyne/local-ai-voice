from __future__ import annotations

from typing import Callable

import numpy as np

from local_ai.slices.voice.shared.audio_processing import (
    NR_IMPORT_ERROR,
    TARGET_SAMPLE_RATE,
    preprocess_audio,
    resample_audio_linear,
)
from local_ai.slices.voice.shared.transcript_policy import should_suppress_transcript, transcribe_chunk
from local_ai.slices.voice.transcribe_live.request import TranscribeLiveRequest
from local_ai.slices.voice.transcribe_live.response import TranscribeLiveResponse


def execute_transcribe_live(
    *,
    request: TranscribeLiveRequest,
    sounddevice_module: object,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
    logger: Callable[[str, bool, float | None], None],
    runtime_error_details: Callable[[Exception], list[str]],
    on_output: Callable[[str], None],
    on_status: Callable[[str], None],
) -> TranscribeLiveResponse:
    if request.chunk_seconds <= 0:
        return TranscribeLiveResponse(exit_code=2, reason="--chunk-seconds must be > 0.")

    if isinstance(sounddevice_module, BaseException):
        return TranscribeLiveResponse(
            exit_code=6,
            reason="Live mode requires sounddevice.",
            details=[f"Import error: {sounddevice_module}", "Install sounddevice or pass a WAV file instead."],
        )

    sd = sounddevice_module
    try:
        default_input = sd.query_devices(kind="input")
        record_sample_rate = int(round(float(default_input["default_samplerate"])))
    except Exception as exc:
        return TranscribeLiveResponse(
            exit_code=7,
            reason="Could not read default input sample rate.",
            details=[f"Runtime error: {exc}", "Check your default microphone device settings."],
        )

    if record_sample_rate <= 0:
        return TranscribeLiveResponse(
            exit_code=7,
            reason="Default input sample rate is invalid.",
            details=[f"Detected: {record_sample_rate}"],
        )

    chunk_samples = int(round(request.chunk_seconds * record_sample_rate))
    if chunk_samples <= 0:
        return TranscribeLiveResponse(
            exit_code=2,
            reason="Computed chunk size is invalid; increase --chunk-seconds.",
        )

    on_status("Live transcription started. Press Ctrl+C to stop.")
    logger(
        f"Recording {record_sample_rate} Hz mono; transcribing every {request.chunk_seconds:.2f}s.",
        request.verbose,
        start,
    )

    try:
        with sd.InputStream(
            samplerate=record_sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        ) as stream:
            chunk_index = 0
            while True:
                data, overflowed = stream.read(chunk_samples)
                chunk_index += 1
                if overflowed:
                    logger("Audio input overflow detected; transcription may skip samples.", request.verbose, start)

                audio = np.asarray(data[:, 0], dtype=np.float32)
                if np.max(np.abs(audio), initial=0.0) < 1e-4:
                    continue
                try:
                    audio = preprocess_audio(audio, record_sample_rate, audio_preprocessor, request.verbose, start, logger)
                except Exception as exc:
                    return TranscribeLiveResponse(
                        exit_code=6,
                        reason="Live audio preprocessing failed.",
                        details=[f"Runtime error: {exc}", NR_IMPORT_ERROR],
                    )
                if request.silence_detect and audio.size == 0:
                    continue
                if record_sample_rate != TARGET_SAMPLE_RATE:
                    audio = resample_audio_linear(audio, record_sample_rate, TARGET_SAMPLE_RATE)

                try:
                    text = transcribe_chunk(pipe, audio, generate_kwargs)
                except Exception as exc:
                    return TranscribeLiveResponse(
                        exit_code=5,
                        reason="Live transcription failed.",
                        details=runtime_error_details(exc),
                    )

                if text and not should_suppress_transcript(text, audio_preprocessor):
                    on_output(f"[chunk {chunk_index}] {text}")
    except KeyboardInterrupt:
        on_status("\nStopped live transcription.")
        return TranscribeLiveResponse(exit_code=0)
    except Exception as exc:
        return TranscribeLiveResponse(
            exit_code=7,
            reason="Failed to capture microphone audio.",
            details=[f"Runtime error: {exc}", "Check microphone permissions and input device availability."],
        )
