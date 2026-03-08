from __future__ import annotations

import argparse

from local_ai.slices.voice.web_ui.server_bootstrap import prepare_server_components


def test_prepare_server_components_validates_and_builds_context() -> None:
    calls: list[str] = []
    args = argparse.Namespace(
        tls_certfile=None,
        tls_keyfile=None,
        chunk_seconds=1.5,
        overlap_seconds=0.0,
    )

    def validate_tls(cert, key) -> None:
        calls.append("tls")

    def validate_chunk(chunk, overlap) -> None:
        calls.append("chunk")

    def create_context_fn(current_args, start_time):
        calls.append(f"context:{start_time}")
        return argparse.Namespace(silence_detect_default=True, vad_mode_default=3)

    def load_index_html_fn(silence_detect_default: bool, vad_mode_default: int) -> str:
        calls.append(f"html:{silence_detect_default}:{vad_mode_default}")
        return "<html></html>"

    def service_factory(*, ctx, index_html):
        calls.append(f"service:{index_html}")
        return {"ctx": ctx, "index_html": index_html}

    ctx, service, start_time = prepare_server_components(
        args=args,
        perf_counter=lambda: 12.5,
        validate_tls=validate_tls,
        validate_chunk=validate_chunk,
        create_context_fn=create_context_fn,
        load_index_html_fn=load_index_html_fn,
        service_factory=service_factory,
    )

    assert start_time == 12.5
    assert service["index_html"] == "<html></html>"
    assert calls == ["tls", "chunk", "context:12.5", "html:True:3", "service:<html></html>"]
