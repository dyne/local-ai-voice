# AGENTS.md

## Scope

Primary public entrypoint:

- `local-ai-voice.py`

Current implementation also relies on these repository entry shells:

- `local_ai_voice.py`
- `browser_webrtc.py`
- `voice_runtime.py`

Canonical implementation now lives under the `local_ai/` package.

Browser UI implementation now also lives under:

- `frontend/`

## Environment

- Use Python `3.11` for install, validation, profiling, packaging, and test commands.

## Current Architecture

The repository is no longer a single-script implementation. It is now organized into a small shared core plus voice-specific vertical slices.

### Functional Shape

- `local_ai/shared/domain`
  - reusable device selection, model resolution, and shared errors
- `local_ai/infrastructure/openvino`
  - OpenVINO Whisper runtime construction and runtime environment setup
- `local_ai/slices/voice`
  - file transcription slice
  - live transcription slice
  - streaming/chunking slice
  - browser/desktop web UI slice
- `frontend`
  - Svelte UI compiled by Vite and served by the Python browser backend when built
- top-level compatibility shells
  - preserve the current CLI, `--server`, and desktop launch behavior

### Design Choices

The current structure follows these rules:

- `VSA`: implement one use-case per slice instead of one giant layer-first module
- `REPR`: keep request/service/response boundaries explicit for each slice
- `Hex, pragmatically applied`: domain rules, infrastructure code, and transport code are separated, but without forcing premature abstraction
- `Light DDD`: keep stable names and invariants close to the code that enforces them

This is intentionally concrete. The repository is voice-first today, with architecture prepared for future OCR, image generation, chat, and agent modules when they become real slices.

### Frontend Design

- Svelte owns browser presentation and page-local state.
- Python still owns session creation, event streaming, websocket audio ingestion, device/model selection, and inference.
- The frontend must consume the existing backend contract instead of re-implementing backend behavior in the browser.
- Prefer Svelte + Vite over SvelteKit because Python remains the only app server.

## Script Intent

- Transcribe WAV files or microphone audio using OpenVINO GenAI Whisper.
- Support device preference via `--device` (`NPU`, `GPU`, `CPU`) and optional device enumeration via `--device list`.
- Prefer OpenVINO static pipeline mode for Whisper execution; CPU may fall back when the plugin does not support `STATIC_PIPELINE`.
- Support `--task`, `--language`, `--timestamps`, `--chunk-seconds`, `--model`, `--offline`, and `--verbose`.
- Default startup opens the local web UI in a `pywebview` desktop window.
- `local-ai-voice.py --server` runs the raw browser server without the desktop wrapper.
- `local-ai-voice.py --cli` forces the non-web CLI path for file or live microphone transcription.
- Auto-download OpenVINO Whisper models from Hugging Face Hub and reuse the shared Hugging Face cache.
- When available, the browser server should serve the built Svelte frontend from `frontend/dist`; otherwise it may fall back to the legacy `web/index.html`.

## Non-Negotiable Behavior

- Resample to `16000 Hz` for Whisper.
- Keep audio processing in mono `float32`.
- In browser mode, keep Opus decode, denoise, and VAD before the final `16000 Hz` resample.
- Keep CLI behavior stable; avoid breaking existing flags.
- Keep error messages concise and diagnostic: what failed and likely reason.
- Keep model execution on OpenVINO GenAI for `NPU`, `GPU`, and `CPU`.
- When `--model` is explicitly provided, do not silently fall back to a different model.
- Print selected device and resolved model path to `stderr` on CLI and browser startup paths.

## CLI Defaults

- `--device` default: `NPU,GPU,CPU`
- `--model` optional; defaults per selected device and may auto-download from Hugging Face if missing:
  - NPU: `OpenVINO/whisper-base.en-int8-ov`
  - GPU/CPU: `OpenVINO/whisper-tiny-fp16-ov`
- `--model` accepts either a local OpenVINO model directory or a Hugging Face repo id in `org/name` form.
- `--offline` disables model downloads and must fail fast if the required model is not already available locally.
- Default mode is desktop web UI; use `--cli` for file/live transcription and `--server` for raw browser-server mode.
- `--chunk-seconds` must be a positive number when running in live mode.
- Browser mode defaults:
  - `--vad-mode 3`
  - `--overlap-seconds 0.0`
- `--profile` enables `py-spy` recording for the current run.
- `--profile-output` optionally sets output SVG path; default is under `profiles/`.

## Model Download And Packaging Behavior

- Reuse the standard shared Hugging Face cache; do not re-download models that are already cached locally.
- Support `HF_TOKEN` when present for authenticated Hugging Face downloads.
- Include `huggingface_hub[hf_xet]` and `hf_xet` in packaging so frozen builds can use Xet-backed model downloads.
- Frozen builds must also bundle dynamic web-server imports needed by browser/server mode, including `uvicorn` and websocket dependencies.
- Frozen builds should eventually package the built frontend assets so desktop/server mode uses the same Svelte UI as source runs.
- `GNUmakefile`, `local-ai-voice.spec`, and the GitHub release workflow should build and package `frontend/dist` before freezing the executable.
- When adding, removing, or changing packaged runtime dependencies, keep all packaging entrypoints in sync:
  - update the checked-in `local-ai-voice.spec`
  - update the PyInstaller flags in `GNUmakefile`
  - update `.github/workflows/build-and-release.yaml`
- In `.github/workflows/build-and-release.yaml`, keep both release jobs aligned:
  - update the `Install build dependencies` step so CI installs the package
  - update the `Generate PyInstaller spec` step so hidden imports, collected submodules, binaries, and data files match the runtime dependency
- If a dependency is imported dynamically at runtime, add the required `--hidden-import`, `--collect-submodules`, `--collect-binaries`, or `--collect-data` entries in the workflow and spec instead of assuming PyInstaller will discover it.

## Architectural Invariants

### Shared

- explicit `--model` must never silently fall back
- offline mode must never download
- cached Hugging Face snapshots should be used before downloading defaults
- resolved model directories must be validated before runtime creation

### Voice

- inference audio remains mono `float32` at `16000 Hz`
- browser decode path remains decode -> denoise -> VAD -> final resample
- live chunking and overlap settings must remain behaviorally stable unless intentionally changed and tested

## Repository Map

Useful current module boundaries:

- `local_ai/shared/domain/devices.py`
- `local_ai/shared/domain/models.py`
- `local_ai/shared/domain/errors.py`
- `local_ai/infrastructure/openvino/whisper.py`
- `local_ai/infrastructure/openvino/runtime_env.py`
- `local_ai/slices/voice/transcribe_file/`
- `local_ai/slices/voice/transcribe_live/`
- `local_ai/slices/voice/transcribe_stream/`
- `local_ai/slices/voice/web_ui/`
- `local_ai/slices/voice/entrypoint.py`
- `local_ai/slices/voice/transcribe_runner.py`
- `frontend/src/App.svelte`
- `frontend/src/lib/runtime-config.js`
- `frontend/src/lib/session-payload.js`

When changing behavior, prefer editing the `local_ai/` package first and keep the top-level scripts as thin adapters.

## Testing Expectations

Use pytest for new code and add unit coverage whenever extracting or changing logic.
Use Vitest for frontend utility modules and browser-side state helpers.

Minimum validation for routine refactors:

1. `py -3.11 -m pytest`
2. `py -3.11 -m py_compile local-ai-voice.py local_ai_voice.py voice_runtime.py browser_webrtc.py`
3. `cd frontend && npm test`
4. `cd frontend && npm run build`

Additional validation by change type:

1. File transcription smoke test:
   - `py -3.11 .\local-ai-voice.py --cli <wav> --model <model_dir>`
2. Live transcription startup test:
   - `py -3.11 .\local-ai-voice.py --cli --model <model_dir> --chunk-seconds 1.0`
3. Device-path check when adjusting device selection:
   - `py -3.11 .\local-ai-voice.py --cli --device CPU --model <cpu_model_dir>`
4. Browser startup test after browser-mode changes:
   - `py -3.11 .\local-ai-voice.py --model <model_dir>`
   - `py -3.11 .\local-ai-voice.py --server --model <model_dir>`
5. Auto-download or offline-path check when adjusting model resolution:
   - `py -3.11 .\local-ai-voice.py --cli <wav> --device CPU --model OpenVINO/whisper-tiny-fp16-ov`
   - `py -3.11 .\local-ai-voice.py --cli <wav> --offline --model <missing_model_dir>`
6. Packaging regression check after frozen-build changes:
   - `py -3.11 -m pytest tests/test_packaging_spec.py`
   - rebuild and run `.\dist\local-ai-voice.exe` when packaging changes matter
7. Frontend selection check after browser frontend changes:
   - ensure `frontend/dist/index.html` is preferred over `web/index.html`
   - run one browser startup path after a frontend build
   - if packaging dependencies changed, verify `.github/workflows/build-and-release.yaml` still matches `local-ai-voice.spec` and `GNUmakefile`

## Profiling

- Install profiler:
  - `py -3.11 -m pip install py-spy`
- Profile file mode:
  - `py -3.11 .\local-ai-voice.py --cli <wav> --model <model_dir> --profile`
- Profile live mode:
  - `py -3.11 .\local-ai-voice.py --cli --model <model_dir> --chunk-seconds 1.0 --profile`
- Profile browser server:
  - `py -3.11 .\local-ai-voice.py --server --model <model_dir> --profile`
- Custom output path:
  - `--profile-output .\profiles\my_run.svg`
- Open the generated SVG in a browser and inspect the hottest stacks first.
