# Architecture

## Status

This document now reflects the current repository state, not just the target design.

The repository has already been refactored from a mostly monolithic voice application into:

- a small shared Local AI core for device and model handling
- an OpenVINO-specific infrastructure layer
- vertical voice slices for file, live, stream, and browser UI flows
- a Svelte frontend that consumes the existing browser/server contracts
- thin compatibility entrypoints that preserve the existing CLI and desktop/server behavior

The current implementation is voice-first. OCR, image generation, chat, and agent slices are planned but not implemented yet.

## Design Summary

The codebase follows four practical architectural rules:

- `VSA`: organize by use-case slices instead of broad layer-first files
- `REPR`: make each transport path explicit through request/service/response style boundaries
- `Hex`: keep domain and orchestration code separated from infrastructure and transport details
- `Light DDD`: use stable names, explicit invariants, and small operational events/decisions close to the domain

The intent is simple reuse:

- shared code decides device selection and model resolution once
- infrastructure code builds OpenVINO runtimes once
- slices own user-facing behavior for one concrete use-case at a time

## Current Topology

```text
local_ai/
  shared/
    domain/
      devices.py
      errors.py
      models.py

  infrastructure/
    openvino/
      runtime_env.py
      whisper.py

  slices/
    voice/
      entrypoint.py
      transcribe_runner.py
      shared/
        audio_processing.py
        transcript_policy.py
      transcribe_file/
        request.py
        response.py
        service.py
      transcribe_live/
        request.py
        response.py
        service.py
      transcribe_stream/
        buffer_decoder.py
        request.py
        response.py
        service.py
      web_ui/
        app_factory.py
        audio_decode.py
        capture_store.py
        chunk_pipeline.py
        event_stream.py
        inference_runner.py
        launch_helpers.py
        launch_modes.py
        message_processor.py
        page_loader.py
        runtime_context.py
        server_bootstrap.py
        server_config.py
        service.py
        session_cleanup.py
        session_decoder.py
        session_registry.py
        session_state.py
        socket_loop.py

local-ai-voice.py
local_ai_voice.py
browser_webrtc.py
voice_runtime.py

frontend/
  package.json
  vite.config.js
  svelte.config.js
  src/
    App.svelte
    main.js
    lib/
      runtime-config.js
      session-payload.js
```

## Current Responsibilities

### Shared Domain

`local_ai/shared/domain` contains the reusable core.

- `devices.py`: device parsing, normalization, availability checks, and selection
- `models.py`: model reference parsing, default model selection, Hugging Face cache reuse, offline/download policy, model directory validation
- `errors.py`: shared domain/runtime-facing exceptions used across the refactor

This layer should stay free of FastAPI, pywebview, WebRTC, and OpenVINO-specific code.

### Infrastructure

`local_ai/infrastructure/openvino` contains OpenVINO-specific runtime bootstrap.

- `whisper.py`: build and configure Whisper/OpenVINO runtime objects
- `runtime_env.py`: runtime environment setup shared by CLI and browser flows

This layer is where provider details belong. The rest of the code should not need to know about low-level OpenVINO setup.

### Voice Slices

`local_ai/slices/voice` contains the current vertical slices.

- `transcribe_file`: file transcription request/response/service flow
- `transcribe_live`: microphone/live transcription request/response/service flow
- `transcribe_stream`: reusable chunk/window preparation for incremental streaming
- `shared`: voice-specific audio invariants and transcript shaping
- `web_ui`: browser transport, session lifecycle, stream decode, chunk orchestration, server bootstrap, and desktop/server launch support

### Compatibility Entrypoints

Top-level scripts still exist to preserve stable behavior:

- `local-ai-voice.py`: public script entrypoint
- `local_ai_voice.py`: CLI argument parsing and dispatch shell
- `browser_webrtc.py`: compatibility facade for browser/server startup
- `voice_runtime.py`: compatibility facade for shared runtime/model/device logic

The goal is to keep these thin and behaviorally stable while the canonical implementation lives under `local_ai/`.

### Frontend

`frontend/` contains the browser UI implementation.

- Svelte owns presentation, browser state, and microphone/session wiring in the page
- Vite builds static assets into `frontend/dist`
- the Python server serves the built app when `frontend/dist/index.html` exists
- the old `web/index.html` remains as a fallback during migration and packaging transitions
- build and release workflows are expected to build `frontend/dist` before packaging the executable
- normal test workflows should treat frontend Vitest and backend pytest as one combined quality gate

## Ubiquitous Language

These terms should stay stable in code and docs:

- `DevicePreference`: ordered user preference such as `NPU,GPU,CPU`
- `DeviceSelection`: resolved device plus detected device inventory
- `ModelReference`: either a local path or a Hugging Face repo id
- `ModelArtifact`: validated local model directory
- `ResolutionPolicy`: offline/download rules
- `Voice Runtime`: initialized Whisper runtime on a selected OpenVINO device
- `Slice`: one concrete use-case with its own request/service/response boundary
- `Session`: browser live-transcription state bound to sockets, buffers, queues, and capture state

## Invariants

### Shared Invariants

- explicit `--model` must never silently fall back
- offline mode must never download
- default models may resolve from the Hugging Face cache before downloading
- resolved model directories must contain the required files
- runtime execution remains on OpenVINO for NPU, GPU, and CPU

### Voice Invariants

- audio must be mono `float32`
- Whisper input must be resampled to `16000 Hz`
- browser flow must remain decode -> denoise -> VAD -> final `16000 Hz` resample
- CLI flags and default modes must remain stable
- startup should print selected device and resolved model path to `stderr`

## How VSA + REPR Show Up In The Current Code

The repository does not yet have a full transport-agnostic port package, but the refactor already applies the pattern in practice.

Examples:

- `transcribe_file/request.py` + `service.py` + `response.py`
- `transcribe_live/request.py` + `service.py` + `response.py`
- `transcribe_stream/request.py` + `service.py` + `response.py`

The top-level transport code builds requests, calls slice services, and formats outputs. This keeps core behavior testable without spinning up the full app.

## Hexagonal Boundary In Practice

The repository currently uses a pragmatic, partial hexagonal split:

- domain modules own validation and resolution rules
- infrastructure modules own OpenVINO runtime construction
- web/CLI modules own transport and startup behavior

The boundary is not yet formalized as `ports/` and `adapters/` packages, but the code is now shaped so that those ports can be introduced incrementally without large rewrites.

That is an intentional design choice: keep the code concrete while extracting stable seams first.

## Browser Architecture

The browser stack has been decomposed into small modules under `local_ai/slices/voice/web_ui`.

Current responsibilities are roughly:

- `app_factory.py`: FastAPI app assembly
- `server_bootstrap.py` and `launch_modes.py`: server and desktop/browser launch setup
- `service.py`: browser session/service shell
- `session_registry.py`, `session_cleanup.py`, `session_state.py`: session lifecycle and registry ownership
- `audio_decode.py`, `session_decoder.py`, `buffer_decoder.py`: streamed audio decode path
- `chunk_pipeline.py`, `inference_runner.py`, `message_processor.py`, `socket_loop.py`: websocket message handling and chunk inference orchestration
- `capture_store.py`: optional WAV capture persistence
- `event_stream.py`: server-sent event queue streaming

The backend browser slice now serves one of two frontend implementations:

- preferred: built Svelte assets from `frontend/dist`
- fallback: legacy `web/index.html`

This is the most complex part of the repository, and the refactor intentionally broke it into narrow, testable seams before changing behavior.

## Frontend Strategy

The frontend migration uses Svelte, not SvelteKit.

That choice is deliberate:

- Python already owns the app server and the runtime-critical backend logic
- the current browser API is already stable enough for a frontend swap
- Svelte provides a cleaner UI/state model without introducing a second server framework

Current frontend contract:

- `POST /session`
- `GET /events/{session_id}`
- `WS /audio/{session_id}`
- `DELETE /session/{session_id}`

The Svelte app is expected to treat those routes as the source of truth and avoid moving runtime logic into the browser.

## Testing State

The refactor has been accompanied by pytest-based unit coverage around the extracted modules.

The test suite currently covers:

- device parsing and selection
- model resolution and Hugging Face cache behavior
- OpenVINO runtime environment setup
- file and live slice services
- stream buffering and chunk preparation
- browser session lifecycle helpers
- browser launch/bootstrap helpers
- browser decode/message/chunk/socket helper flows
- frontend runtime-config and session-payload helpers through Vitest
- frontend selection of built assets over legacy HTML
- packaging/spec regression checks for frozen builds

The architecture depends on preserving this test-first approach for future slices.

## Current Design Choices

### Why Keep Compatibility Facades

The project still exposes the historical scripts because the user-facing interface is already useful and should not churn while internals are refactored.

### Why Voice-First

The only implemented capability is voice, so the architecture stays concrete and optimized around real behavior. OCR, image, chat, and agent support should be added only when the first real slice exists.

### Why No Heavy DI Framework

The repository is still small enough that explicit imports and thin wrappers are easier to maintain than a container framework.

### Why HTMX Is Only A Future Partial Fit

HTMX is a good fit for server-rendered upload/result flows and diagnostics pages, but not for microphone capture and websocket/WebRTC audio transport. Live browser transcription still needs focused JavaScript.

### Why Svelte Fits Now

The backend browser contract is now narrow enough that the old inline HTML/JavaScript page can be replaced by a compiled frontend without changing inference or transport behavior. Svelte improves maintainability on the browser side while leaving device/model/runtime concerns in Python.

## Next Steps

The next architectural steps should be incremental:

1. Keep shrinking top-level compatibility files as behavior stays stable.
2. Complete the Svelte migration and remove the legacy inline browser page once packaging and startup paths are stable.
3. Introduce explicit application ports only when at least two slices benefit from the abstraction.
4. Add the next capability as a real vertical slice, likely OCR or chat.
5. Add a lightweight bootstrap/container only when multiple capabilities need shared wiring.
6. Move toward a unified `local-ai` CLI entrypoint once voice compatibility no longer depends on the legacy script layout.

## What To Avoid

- reintroducing large monolithic utility modules
- abstract interfaces without a second real consumer
- mixing transport, runtime construction, and domain validation in one file
- changing CLI/browser behavior during structural refactors unless covered and intentional
- building generic future capability scaffolding before a real use-case exists
