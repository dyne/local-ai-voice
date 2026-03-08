# AGENT.md

## Scope
Applies to the single script:
- `local-ai-voice.py`

## Environment
- Use Python `3.11` for install, validation, profiling, and packaging commands.

## Script Intent
- Transcribe WAV files or microphone audio using OpenVINO GenAI Whisper.
- Supports device preference via `--device` (`NPU`, `GPU`, `CPU`) and optional device enumeration via `--device list`.
- Prefers OpenVINO static pipeline mode for Whisper execution; CPU may fall back when the plugin does not support `STATIC_PIPELINE`.
- Supports `--task`, `--language`, `--timestamps`, `--chunk-seconds`, `--model`, `--offline`, and `--verbose`.
- Default startup opens the local web UI in a `pywebview` desktop window.
- `local-ai-voice.py --server` runs the raw browser server without the desktop wrapper.
- `local-ai-voice.py --cli` forces the non-web CLI path for file or live microphone transcription.
- Can auto-download OpenVINO Whisper models from Hugging Face Hub and reuse the shared Hugging Face cache.

## Non-Negotiable Behavior
- Resample to `16000 Hz` for Whisper.
- Keep audio processing in mono `float32`.
- In browser mode, keep Opus decode, denoise, and VAD before the final `16000 Hz` resample.
- Keep CLI behavior stable; avoid breaking existing flags.
- Keep error messages concise and diagnostic (what failed + likely reason).
- Keep model execution on OpenVINO GenAI for `NPU`, `GPU`, and `CPU`.
- When `--model` is explicitly provided, do not silently fall back to a different model.

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
- `--profile-output` optionally sets output SVG path (default under `profiles/`).
- CLI and browser startup should print the selected device and resolved model path to `stderr`.

## Model Download Behavior
- Reuse the standard shared Hugging Face cache; do not re-download models that are already cached locally.
- Support `HF_TOKEN` when present for authenticated Hugging Face downloads.
- Include `huggingface_hub[hf_xet]`/`hf_xet` in packaging so frozen builds can use Xet-backed model downloads.

# Validation Checklist
1. `py -3.11 -m py_compile local-ai-voice.py local_ai_voice.py voice_runtime.py browser_webrtc.py`
2. File transcription smoke test:
   - `py -3.11 .\local-ai-voice.py --cli <wav> --model <model_dir>`
3. Live transcription startup test:
   - `py -3.11 .\local-ai-voice.py --cli --model <model_dir> --chunk-seconds 1.0`
4. Run one device-path check when adjusting device selection:
   - `py -3.11 .\local-ai-voice.py --cli --device CPU --model <cpu_model_dir>`
5. Browser startup test after browser-mode changes:
   - `py -3.11 .\local-ai-voice.py --model <model_dir>`
   - `py -3.11 .\local-ai-voice.py --server --model <model_dir>`
6. Run one auto-download or offline-path check when adjusting model resolution:
   - `py -3.11 .\local-ai-voice.py --cli <wav> --device CPU --model OpenVINO/whisper-tiny-fp16-ov`
   - `py -3.11 .\local-ai-voice.py --cli <wav> --offline --model <missing_model_dir>`

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
- Open the generated SVG in a browser and focus on hottest stacks first.
