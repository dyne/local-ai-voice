# AGENT.md

## Scope
Applies to the single script:
- `local-ai-voice.py`

## Environment
- Use Python `3.11` for install, validation, profiling, and packaging commands.

## Script Intent
- Transcribe WAV files or microphone audio using OpenVINO GenAI Whisper.
- Supports device preference via `--device` (`NPU`, `GPU`, `CPU`) and optional device enumeration via `--device list`.
- Uses OpenVINO static pipeline mode for Whisper execution.
- Supports `--task`, `--language`, `--timestamps`, `--chunk-seconds`, `--model`, and `--verbose`.
- Browser mode is served through `local-ai-voice.py --web` and accepts Opus over WebSocket from the browser.

## Non-Negotiable Behavior
- Resample to `16000 Hz` for Whisper.
- Keep audio processing in mono `float32`.
- In browser mode, keep Opus decode, denoise, and VAD before the final `16000 Hz` resample.
- Keep CLI behavior stable; avoid breaking existing flags.
- Keep error messages concise and diagnostic (what failed + likely reason).

## CLI Defaults
- `--device` default: `NPU,GPU,CPU`
- `--model` optional; defaults per selected device:
  - NPU: `whisper-base.en-int8-ov`
  - GPU/CPU: `whisper-tiny-fp16-ov`
- `--chunk-seconds` must be a positive number when running in live mode.
- Browser mode defaults:
  - `--vad-mode 3`
  - `--overlap-seconds 0.0`
- `--profile` enables `py-spy` recording for the current run.
- `--profile-output` optionally sets output SVG path (default under `profiles/`).

# Validation Checklist
1. `py -3.11 -m py_compile local-ai-voice.py`
2. File transcription smoke test:
   - `py -3.11 .\local-ai-voice.py <wav> --model <model_dir>`
3. Live transcription startup test:
   - `py -3.11 .\local-ai-voice.py --model <model_dir> --chunk-seconds 1.0`
4. Run one device-path check when adjusting device selection:
   - `py -3.11 .\local-ai-voice.py --device CPU --model <cpu_model_dir>`
5. Browser startup test after browser-mode changes:
   - `py -3.11 .\local-ai-voice.py --web --model <model_dir>`

## Profiling
- Install profiler:
  - `py -3.11 -m pip install py-spy`
- Profile file mode:
  - `py -3.11 .\local-ai-voice.py <wav> --model <model_dir> --profile`
- Profile live mode:
  - `py -3.11 .\local-ai-voice.py --model <model_dir> --chunk-seconds 1.0 --profile`
- Profile browser server:
  - `py -3.11 .\local-ai-voice.py --web --model <model_dir> --profile`
- Custom output path:
  - `--profile-output .\profiles\my_run.svg`
- Open the generated SVG in a browser and focus on hottest stacks first.
