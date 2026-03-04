# AGENT.md

## Scope
Applies to the single script:
- `transcribe_wav.py`

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

# Validation Checklist
1. `python -m py_compile transcribe_wav.py`
2. File transcription smoke test:
   - `python .\transcribe_wav.py <wav> --model <model_dir>`
3. Live transcription startup test:
   - `python .\transcribe_wav.py --model <model_dir> --chunk-seconds 1.0`
4. Run one device-path check when adjusting device selection:
   - `python .\transcribe_wav.py --device CPU --model <cpu_model_dir>`

