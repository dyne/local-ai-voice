from __future__ import annotations

import asyncio

import pytest

from local_ai.slices.voice.web_ui.inference_runner import run_chunk_inference


@pytest.mark.anyio
async def test_run_chunk_inference_returns_text() -> None:
    async def fake_to_thread(fn, *args):
        return "hello"

    result = await run_chunk_inference(
        chunk=[0.1, 0.2],
        pipe=object(),
        generate_kwargs={},
        audio_preprocessor=None,
        infer_lock=asyncio.Lock(),
        transcribe_fn=lambda pipe, chunk, kwargs: "hello",
        should_suppress_fn=lambda text, preprocessor: False,
        likely_reason_details_fn=lambda exc: [f"Runtime error: {exc}"],
        to_thread_fn=fake_to_thread,
    )

    assert result.text == "hello"
    assert result.error is None


@pytest.mark.anyio
async def test_run_chunk_inference_suppresses_transcript() -> None:
    async def fake_to_thread(fn, *args):
        return "uh"

    result = await run_chunk_inference(
        chunk=[0.1, 0.2],
        pipe=object(),
        generate_kwargs={},
        audio_preprocessor=object(),
        infer_lock=asyncio.Lock(),
        transcribe_fn=lambda pipe, chunk, kwargs: "uh",
        should_suppress_fn=lambda text, preprocessor: True,
        likely_reason_details_fn=lambda exc: [f"Runtime error: {exc}"],
        to_thread_fn=fake_to_thread,
    )

    assert result.text is None
    assert result.error is None


@pytest.mark.anyio
async def test_run_chunk_inference_returns_error_message() -> None:
    async def fake_to_thread(fn, *args):
        raise RuntimeError("boom")

    result = await run_chunk_inference(
        chunk=[0.1, 0.2],
        pipe=object(),
        generate_kwargs={},
        audio_preprocessor=None,
        infer_lock=asyncio.Lock(),
        transcribe_fn=lambda pipe, chunk, kwargs: "unused",
        should_suppress_fn=lambda text, preprocessor: False,
        likely_reason_details_fn=lambda exc: [f"Runtime error: {exc}", "detail"],
        to_thread_fn=fake_to_thread,
    )

    assert result.text is None
    assert result.error == "[server error] Live transcription failed: Runtime error: boom"
