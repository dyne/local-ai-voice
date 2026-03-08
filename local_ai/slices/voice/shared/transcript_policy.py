from __future__ import annotations

import numpy as np

from local_ai.slices.voice.shared.audio_processing import AudioPreprocessor, VAD_FRAME_MS

WEAK_SPEECH_RATIO_MARGIN = 0.08
WEAK_SPEECH_FRAMES_MARGIN = 1
WEAK_UTTERANCE_MS_MARGIN = 60
COMMON_WEAK_HALLUCINATIONS = {
    ".",
    ",",
    "?",
    "!",
    "...",
    "you",
    "uh",
    "um",
    "hmm",
    "hm",
}


def result_to_text(result: object) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    texts = getattr(result, "texts", None)
    if isinstance(texts, list) and texts:
        return str(texts[0])
    return str(result)


def transcribe_chunk(pipe: object, audio: np.ndarray, generate_kwargs: dict[str, object]) -> str:
    result = pipe.generate(audio.tolist(), **generate_kwargs)
    return result_to_text(result).strip()


def _is_common_weak_hallucination(text: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return False
    if normalized in COMMON_WEAK_HALLUCINATIONS:
        return True
    if all(ch in ".,!?:;-'\"()[]{} " for ch in normalized):
        return True
    return False


def should_suppress_transcript(text: str, preprocessor: object | None) -> bool:
    if preprocessor is None or not isinstance(preprocessor, AudioPreprocessor):
        return False
    if not text:
        return False
    if not _is_common_weak_hallucination(text):
        return False
    if preprocessor.last_total_frames <= 0:
        return False
    speech_ratio = preprocessor.last_speech_frames / float(preprocessor.last_total_frames)
    min_duration_frames = max(1, int(np.ceil(preprocessor.min_utterance_ms / float(VAD_FRAME_MS))))
    weak_ratio = speech_ratio <= (preprocessor.min_speech_ratio + WEAK_SPEECH_RATIO_MARGIN)
    weak_frames = preprocessor.last_speech_frames <= (min_duration_frames + WEAK_SPEECH_FRAMES_MARGIN)
    weak_run = preprocessor.last_max_run <= (preprocessor.min_speech_frames + WEAK_SPEECH_FRAMES_MARGIN)
    weak_duration = (preprocessor.last_speech_frames * VAD_FRAME_MS) <= (
        preprocessor.min_utterance_ms + WEAK_UTTERANCE_MS_MARGIN
    )
    return preprocessor.last_was_hangover or ((weak_ratio and weak_frames) or (weak_run and weak_duration))


def setup_error_exit_code(reason: str) -> int:
    if reason.startswith("Model directory not found:"):
        return 2
    if reason.startswith("Failed to create WhisperPipeline on "):
        return 4
    return 3
