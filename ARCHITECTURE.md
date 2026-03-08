# Architecture

## Goals

- Keep device detection, model resolution, cache reuse, downloads, and runtime bootstrap in one reusable core.
- Build user-facing capabilities as vertical slices on top of that core.
- Preserve current voice behavior while creating a stable path for OCR, image generation, chat, and agentic workflows.
- Prefer simple explicit boundaries over generic frameworks.

## Architectural Principles

- `VSA`: organize by use-case slice, not by technical layer alone.
- `REPR`: each CLI command or HTTP route has its own request, endpoint, and response contract.
- `Hex`: domain and application logic depend on ports; infrastructure provides adapters.
- `Light DDD`: use shared domain language, enforce invariants near the domain, and emit small operational domain events.
- `HTMX first`: use server-rendered HTML and HTMX where browser APIs are not required; use focused JavaScript only for cases like microphone/WebRTC streaming.

## Target Topology

```text
local_ai/
  shared/
    domain/
      devices.py
      models.py
      runtimes.py
      errors.py
      events.py
      types.py
    application/
      ports/
        device_catalog.py
        model_store.py
        runtime_factory.py
        event_bus.py
        profiler.py
        clock.py
    infrastructure/
      openvino/
        adapters/
          device_catalog.py
          model_store.py
          whisper_runtime_factory.py
          llm_runtime_factory.py
          diffusion_runtime_factory.py
          ocr_runtime_factory.py
      local/
        adapters/
          stderr_event_bus.py
          pyspy_profiler.py
    bootstrap/
      container.py
      settings.py

  slices/
    voice/
      transcribe_file/
        domain.py
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_http.py
      transcribe_live/
        domain.py
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_ws.py
      web_ui/
        endpoint_http.py
        templates/
      shared/
        audio_processing.py
        transcript_policy.py

    ocr/
      recognize_image/
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_http.py

    image/
      generate_image/
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_http.py

    chat/
      generate_text/
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_http.py
      chat_session/
        request.py
        response.py
        service.py
        endpoint_http.py

    agent/
      run_agent/
        request.py
        response.py
        service.py
        endpoint_cli.py
        endpoint_http.py
      tool_invocation/
        request.py
        response.py
        service.py

app/
  cli.py
  http.py
```

## Ubiquitous Language

These names should stay stable across code, docs, logs, and CLI output.

- `Capability`: a concrete model-backed use-case such as `voice.whisper`, `chat.llm`, `ocr.vision`, `image.diffusion`.
- `DevicePreference`: ordered device intent such as `NPU,GPU,CPU`.
- `DeviceSelection`: selected device plus the detected device inventory.
- `ModelReference`: either a local model path or a Hugging Face repo id.
- `ModelArtifact`: a validated local model directory ready for runtime construction.
- `ResolutionPolicy`: rules such as offline mode and whether download is allowed.
- `RuntimeSession`: an initialized provider runtime bound to a capability, device, and model artifact.
- `InferenceRequest`: normalized input for one capability.
- `InferenceResult`: normalized output for one capability.

## Shared Invariants

- Explicit `--model` never silently falls back to another model.
- Offline mode never downloads.
- Missing default models may be resolved from the shared Hugging Face cache before downloading.
- A model artifact is only valid if it contains the required files for its capability.
- OpenVINO remains the runtime backend for NPU, GPU, and CPU execution.

### Voice Invariants

- Always resample to `16000 Hz` before Whisper inference.
- Always use mono `float32` audio for inference.
- Browser audio flow remains decode -> denoise -> VAD -> final `16000 Hz` resample.
- CLI flags and error semantics remain stable during migration.

## Hexagonal Boundaries

### Core Ports

```python
class DeviceCatalogPort(Protocol):
    def list_available(self) -> list[str]: ...
    def select(self, preference: str) -> DeviceSelection: ...


class ModelStorePort(Protocol):
    def resolve(self, capability: str, reference: ModelReference | None, device: str, policy: ResolutionPolicy) -> ModelArtifact: ...


class RuntimeFactoryPort(Protocol):
    def create(self, capability: str, artifact: ModelArtifact, device: DeviceSelection) -> RuntimeSession: ...


class EventBusPort(Protocol):
    def publish(self, event: DomainEvent) -> None: ...
```

### Adapters

- OpenVINO device enumeration adapter
- Hugging Face cache/download model store adapter
- OpenVINO runtime factories per capability
- stderr/local profiling/event adapters

The domain and application layers must not import FastAPI, pywebview, WebRTC, Hugging Face Hub, or OpenVINO directly.

## VSA + REPR Slice Pattern

Each use-case slice owns:

- `request.py`: validated input DTOs
- `response.py`: explicit output DTOs
- `service.py`: orchestration using ports
- `endpoint_cli.py` and/or `endpoint_http.py`: transport-specific glue
- `domain.py`: capability-specific value objects and invariants when needed

Example:

```text
slices/voice/transcribe_file/
  request.py
  response.py
  service.py
  endpoint_cli.py
```

The endpoint builds a request object, calls the service, and returns a response object. Transport code stays out of domain logic.

## Domain Events

Use small operational events for observability and future UI hooks:

- `DeviceSelected`
- `ModelResolvedFromCache`
- `ModelDownloaded`
- `RuntimeCreated`
- `InferenceStarted`
- `InferenceCompleted`
- `InferenceFailed`

This is not event sourcing. Events exist for diagnostics, profiling, and incremental UX improvements.

## UI Strategy

Use HTMX for:

- OCR upload/result pages
- chat request/response forms
- image generation forms and polling
- diagnostics/model management pages

Use focused JavaScript where browser APIs force it:

- microphone capture
- MediaRecorder / WebRTC
- websocket audio streaming

Voice live transcription should keep the current JS capture path; surrounding UI can still move toward server-rendered pages and HTMX-backed partials later.

## CLI and HTTP Shape

Long term, move toward one unified entrypoint:

```text
local-ai voice ...
local-ai voice --server ...
local-ai ocr ...
local-ai image ...
local-ai chat ...
local-ai agent ...
local-ai devices list
local-ai models resolve ...
```

Common flags:

- `--device`
- `--model`
- `--offline`
- `--verbose`
- `--profile`

Capability-specific flags remain within each slice.

Suggested HTTP examples:

- `GET /voice`
- `POST /voice/transcribe-file`
- `WS /voice/live/{session_id}`
- `POST /ocr/recognize`
- `POST /chat/message`
- `POST /image/generate`
- `GET /image/jobs/{id}`

## Migration Strategy

### Phase 1: Extract the Shared Spine

- Create `local_ai/shared/domain` for errors, devices, and models.
- Create `local_ai/infrastructure/openvino` for Whisper runtime construction.
- Turn `voice_runtime.py` into a compatibility layer that re-exports from the new package.

### Phase 2: Slice the Voice Application

- Split voice into `transcribe_file` and `transcribe_live` slices.
- Move audio preprocessing into `slices/voice/shared`.
- Keep current CLI/web entrypoints as thin adapters while preserving flags and behavior.

### Phase 3: Introduce Bootstrap

- Add a lightweight dependency container in `bootstrap/container.py`.
- Resolve ports to adapters in one place.
- Keep wiring explicit and local.

### Phase 4: Add New Capabilities

- Add OCR, image generation, chat, and agent slices one by one.
- Reuse the same ports for device selection, model resolution, and runtime creation.
- Add provider adapters only when a capability actually exists.

## What To Avoid

- A giant shared `utils.py`
- One generic `InferenceService` for all tasks
- Deep inheritance trees
- Domain services importing transport or provider libraries
- Abstracting future capabilities before their first real use-case exists

## Current First Refactor

The first concrete refactor for this repository is:

1. Save this architecture document.
2. Extract device selection, model resolution, and Whisper runtime creation into the new `local_ai` package.
3. Keep `local-ai-voice.py`, [local_ai_voice.py](/abs/path/C:/Users/denis/devel/local-ai-voice/local_ai_voice.py), [browser_webrtc.py](/abs/path/C:/Users/denis/devel/local-ai-voice/browser_webrtc.py), and [voice_runtime.py](/abs/path/C:/Users/denis/devel/local-ai-voice/voice_runtime.py) behaviorally stable while rewiring imports.
4. Validate with `py_compile` and targeted runtime resolution checks before continuing into deeper slice extraction.
