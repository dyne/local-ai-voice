# Local-AI-voice

Voice dictation tool that converts speech to text using AI models running
locally on your machine. It is designed to work fully offline: audio is
processed on-device, without sending data to external services.

This tool uses OpenVINO GenAI Whisper and takes advantage of modern hardware
acceleration such as NPUs or GPUs when available, while remaining usable on
standard CPUs.

The focus is on privacy, autonomy, and predictable behaviour: your audio never
leaves the machine, models are locally managed, and the software can be
integrated into scripts, editors, or command line workflows.

# Usage

## Python Version

Use Python `3.11`. The Makefile defaults to `py -3.11`, and the audio
dependencies in this repo are expected to be installed against that
interpreter.

Browser streaming requires:
- `websockets` for `uvicorn` WebSocket support
- `av` for decoding browser Opus chunks on the server

## Install dependencies

```sh
py -3.11 -m venv .venv
.venv\Scripts\activate
make install
```

## Run local transcription

Transcribe a WAV file:

```sh
py -3.11 ./local-ai-voice.py input.wav --model ./whisper-tiny-fp16-ov
```

Run live microphone transcription:

```sh
py -3.11 ./local-ai-voice.py --model ./whisper-tiny-fp16-ov --chunk-seconds 1.0
```

Noise reduction and WebRTC VAD speech gating are enabled by default. Disable them with:

```sh
py -3.11 ./local-ai-voice.py --no-silence-detect --model ./whisper-tiny-fp16-ov input.wav
```

## Run browser transcription

Start the local browser transcription server:

```sh
make run-web
```

Equivalent direct command:

```sh
py -3.11 ./local-ai-voice.py --web --model ./whisper-tiny-fp16-ov
```

The browser UI captures microphone audio in the browser and streams Opus over a
WebSocket connection to the local server. The server decodes Opus, runs noise
reduction and VAD at the decoded sample rate, and only resamples to `16 kHz`
immediately before Whisper inference.

Browser defaults:
- browser DSP is off by default:
  - `Echo cancellation`
  - `Noise suppression`
  - `Auto gain control`
- `Voice enhance` is on by default
- `VAD` default is `3`
- overlap default is `0.00s`

Notes:
- Chromium commonly uses `audio/webm;codecs=opus`
- `Save WAV capture` records the exact `16 kHz` mono audio sent to Whisper
- `Client debug` controls browser-side and server-side debug messages shown in the page

## Build standalone executable

Build the unified executable:

```sh
make build
```

The resulting binary supports both modes:

```sh
.\dist\local-ai-voice.exe input.wav --model .\whisper-tiny-fp16-ov
.\dist\local-ai-voice.exe --web --model .\whisper-tiny-fp16-ov
```

# License

Local-AI-voice is Copyright (C) 2026 by the Dyne.org Foundation

It is distributed under the Affero GNU General Public License v3
