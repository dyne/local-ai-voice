from __future__ import annotations

from typing import Callable


def prepare_server_components(
    *,
    args: object,
    perf_counter: Callable[[], float],
    validate_tls: Callable[[object, object], None],
    validate_chunk: Callable[[float, float], None],
    create_context_fn: Callable[[object, float], object],
    load_index_html_fn: Callable[[bool, int], str],
    service_factory: Callable[..., object],
) -> tuple[object, object, float]:
    validate_tls(getattr(args, "tls_certfile"), getattr(args, "tls_keyfile"))
    validate_chunk(float(getattr(args, "chunk_seconds")), float(getattr(args, "overlap_seconds")))
    start_time = perf_counter()
    ctx = create_context_fn(args, start_time)
    service = service_factory(
        ctx=ctx,
        index_html=load_index_html_fn(getattr(ctx, "silence_detect_default"), getattr(ctx, "vad_mode_default")),
    )
    return ctx, service, start_time
