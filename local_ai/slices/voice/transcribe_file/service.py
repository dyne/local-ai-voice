from __future__ import annotations

from typing import Callable

from local_ai.slices.voice.shared.audio_processing import (
    NR_IMPORT_ERROR,
    ensure_sample_rate,
    preprocess_audio,
    read_wav_mono_float32,
)
from local_ai.slices.voice.shared.transcript_policy import should_suppress_transcript, transcribe_chunk
from local_ai.slices.voice.transcribe_file.request import TranscribeFileRequest
from local_ai.slices.voice.transcribe_file.response import TranscribeFileResponse


def execute_transcribe_file(
    *,
    request: TranscribeFileRequest,
    pipe: object,
    audio_preprocessor: object | None,
    generate_kwargs: dict[str, object],
    start: float,
    logger: Callable[[str, bool, float | None], None],
    runtime_error_details: Callable[[Exception], list[str]],
) -> TranscribeFileResponse:
    if not request.wav_path.exists():
        return TranscribeFileResponse(exit_code=2, reason=f"Input file not found: {request.wav_path}")

    logger("Reading WAV", request.verbose, start)
    audio, sample_rate = read_wav_mono_float32(request.wav_path)
    try:
        audio = preprocess_audio(audio, sample_rate, audio_preprocessor, request.verbose, start, logger)
    except Exception as exc:
        return TranscribeFileResponse(
            exit_code=6,
            reason="Audio preprocessing failed.",
            details=[f"Runtime error: {exc}", NR_IMPORT_ERROR],
        )

    if audio.size == 0:
        return TranscribeFileResponse(
            exit_code=2,
            reason="No speech detected.",
            details=["Input appears silent after noise reduction."],
        )

    audio, sample_rate = ensure_sample_rate(audio, sample_rate, request.verbose, start, logger)
    logger(f"Audio prepared: samples={audio.shape[0]}, sample_rate={sample_rate} Hz", request.verbose, start)

    try:
        text = transcribe_chunk(pipe, audio, generate_kwargs)
    except Exception as exc:
        return TranscribeFileResponse(
            exit_code=5,
            reason="Transcription failed.",
            details=runtime_error_details(exc),
        )

    if should_suppress_transcript(text, audio_preprocessor):
        return TranscribeFileResponse(
            exit_code=2,
            reason="No speech detected.",
            details=["Input appears to contain only weak non-speech noise."],
        )

    return TranscribeFileResponse(exit_code=0, text=text)
