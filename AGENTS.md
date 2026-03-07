# AGENT.md

## Scope
Applies to the single script:
- `local-ai-voice.py`

## Script Intent
- Transcribe WAV files or microphone audio using OpenVINO GenAI Whisper.
- Supports device preference via `--device` (`NPU`, `GPU`, `CPU`) and optional device enumeration via `--device list`.
- Uses OpenVINO static pipeline mode for Whisper execution.
- Supports `--task`, `--language`, `--timestamps`, `--chunk-seconds`, `--model`, and `--verbose`.

## Non-Negotiable Behavior
- Resample to `16000 Hz` for Whisper.
- Keep audio processing in mono `float32`.
- Keep CLI behavior stable; avoid breaking existing flags.
- Keep error messages concise and diagnostic (what failed + likely reason).

## CLI Defaults
- `--device` default: `NPU,GPU,CPU`
- `--model` optional; defaults per selected device:
  - NPU: `whisper-base.en-int8-ov`
  - GPU/CPU: `whisper-tiny-fp16-ov`
- `--chunk-seconds` must be a positive number when running in live mode.
- `--profile` enables `py-spy` recording for the current run.
- `--profile-output` optionally sets output SVG path (default under `profiles/`).

# Validation Checklist
1. `python -m py_compile local-ai-voice.py`
2. File transcription smoke test:
   - `python .\local-ai-voice.py <wav> --model <model_dir>`
3. Live transcription startup test:
   - `python .\local-ai-voice.py --model <model_dir> --chunk-seconds 1.0`
4. Run one device-path check when adjusting device selection:
   - `python .\local-ai-voice.py --device CPU --model <cpu_model_dir>`

## Profiling
- Install profiler:
  - `python -m pip install py-spy`
- Profile file mode:
  - `python .\local-ai-voice.py <wav> --model <model_dir> --profile`
- Profile live mode:
  - `python .\local-ai-voice.py --model <model_dir> --chunk-seconds 1.0 --profile`
- Profile WebRTC server:
  - `python .\browser_webrtc.py --model <model_dir> --profile`
- Custom output path:
  - `--profile-output .\profiles\my_run.svg`
- Open the generated SVG in a browser and focus on hottest stacks first.
