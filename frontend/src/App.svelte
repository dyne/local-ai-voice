<script>
  import { onDestroy } from "svelte";

  import { readRuntimeConfig } from "./lib/runtime-config";
  import {
    buildSessionPayload,
    CHUNK_TIMESLICE_MS,
    preferredMimeType,
    TARGET_AUDIO_BITRATE,
    wsBaseUrl,
  } from "./lib/session-payload";

  const runtimeConfig = readRuntimeConfig(window);

  let saveSample = false;
  let silenceDetect = runtimeConfig.silenceDetectDefault;
  let echoCancellation = false;
  let noiseSuppression = false;
  let autoGainControl = false;
  let clientDebug = false;
  let vadMode = String(runtimeConfig.vadModeDefault);
  let chunkSeconds = "1.5";
  let overlapSeconds = "0.00";
  let status = "Idle";
  let transcriptLines = [];

  let stream = null;
  let eventSource = null;
  let audioSocket = null;
  let mediaRecorder = null;
  let sessionId = null;
  let sentBlobCount = 0;

  function appendLine(value) {
    transcriptLines = [...transcriptLines, value];
  }

  function appendClientDebug(value) {
    if (clientDebug) {
      appendLine(value);
    }
  }

  function setStatus(value) {
    status = value;
  }

  async function start() {
    sessionId = globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : String(Date.now());
    setStatus("Requesting microphone...");

    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 48000,
        echoCancellation,
        noiseSuppression,
        autoGainControl,
      },
      video: false,
    });

    const mimeType = preferredMimeType();
    if (!mimeType) {
      throw new Error("No supported Opus MediaRecorder MIME type");
    }

    const trackSettings = stream.getAudioTracks()[0]?.getSettings?.() ?? {};
    appendClientDebug(
      `[client] mime=${mimeType} sampleRate=${trackSettings.sampleRate || "unknown"} channelCount=${trackSettings.channelCount || "unknown"}`,
    );

    const response = await fetch("/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        buildSessionPayload({
          sessionId,
          saveSample,
          silenceDetect,
          debug: clientDebug,
          vadMode,
          chunkSeconds,
          overlapSeconds,
          mimeType,
          audioBitrate: TARGET_AUDIO_BITRATE,
        }),
      ),
    });
    if (!response.ok) {
      throw new Error(`Session setup failed: ${await response.text()}`);
    }

    eventSource = new EventSource(`/events/${sessionId}`);
    eventSource.onmessage = (event) => appendLine(event.data);
    eventSource.onerror = () => setStatus("Event stream disconnected");

    audioSocket = new WebSocket(`${wsBaseUrl()}/audio/${sessionId}`);
    await new Promise((resolve, reject) => {
      audioSocket.onopen = resolve;
      audioSocket.onerror = () => reject(new Error("Audio socket failed to open"));
    });
    appendClientDebug("[client] audio websocket open");

    mediaRecorder = new MediaRecorder(stream, {
      mimeType,
      audioBitsPerSecond: TARGET_AUDIO_BITRATE,
    });
    appendClientDebug(`[client] recorder=${mediaRecorder.mimeType || mimeType} bitrate=${TARGET_AUDIO_BITRATE}`);
    mediaRecorder.ondataavailable = async (event) => {
      if (!event.data || event.data.size === 0) {
        if (sentBlobCount < 4) {
          appendClientDebug("[client] empty recorder blob");
        }
        return;
      }
      if (audioSocket?.readyState === WebSocket.OPEN) {
        sentBlobCount += 1;
        if (sentBlobCount <= 4) {
          appendClientDebug(`[client] blob #${sentBlobCount} size=${event.data.size}`);
        }
        audioSocket.send(await event.data.arrayBuffer());
      }
    };
    mediaRecorder.onerror = (event) => {
      appendClientDebug(`[client error] recorder ${String(event.error || event.name || event)}`);
    };
    mediaRecorder.start(CHUNK_TIMESLICE_MS);

    const actualRate = trackSettings.sampleRate ? `${trackSettings.sampleRate} Hz` : "native rate";
    setStatus(`Streaming Opus (${mimeType}, ${TARGET_AUDIO_BITRATE / 1000} kbps, ${actualRate})`);
  }

  async function stop() {
    setStatus("Stopping...");

    if (mediaRecorder) {
      try {
        mediaRecorder.ondataavailable = null;
        mediaRecorder.onerror = null;
        if (mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        }
      } catch {
        // Ignore browser shutdown noise.
      }
      mediaRecorder = null;
    }
    if (audioSocket) {
      try {
        audioSocket.close();
      } catch {
        // Ignore browser shutdown noise.
      }
      audioSocket = null;
    }
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (sessionId) {
      try {
        await fetch(`/session/${sessionId}`, { method: "DELETE" });
      } catch {
        // Ignore best-effort cleanup failures.
      }
    }
    sessionId = null;
    sentBlobCount = 0;
    setStatus("Idle");
  }

  onDestroy(() => {
    void stop();
  });
</script>

<svelte:head>
  <title>Local AI Voice</title>
</svelte:head>

<div class="shell">
  <main class="card">
    <section class="hero">
      <p class="eyebrow">Local AI Voice</p>
      <h1>Browser microphone transcription over the existing Python voice runtime.</h1>
      <p class="lede">
        The UI is now Svelte-based, but device selection, model resolution, streaming decode,
        VAD, and OpenVINO inference still run on the same backend contracts.
      </p>
    </section>

    <section class="controls">
      <div class="button-row">
        <button class="start" on:click={() => void start()} disabled={Boolean(stream)}>Start</button>
        <button class="stop" on:click={() => void stop()} disabled={!stream}>Stop</button>
        <span class="status">{status}</span>
      </div>

      <div class="grid">
        <label><input type="checkbox" bind:checked={saveSample} /> Save WAV capture</label>
        <label><input type="checkbox" bind:checked={silenceDetect} /> Voice enhance</label>
        <label><input type="checkbox" bind:checked={echoCancellation} /> Echo cancellation</label>
        <label><input type="checkbox" bind:checked={noiseSuppression} /> Noise suppression</label>
        <label><input type="checkbox" bind:checked={autoGainControl} /> Auto gain control</label>
        <label><input type="checkbox" bind:checked={clientDebug} /> Client debug</label>
        <label>
          VAD
          <select bind:value={vadMode}>
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
          </select>
        </label>
        <label>
          Chunk (s)
          <input bind:value={chunkSeconds} type="number" min="0.2" step="0.1" />
        </label>
        <label>
          Overlap (s)
          <input bind:value={overlapSeconds} type="number" min="0.0" step="0.05" />
        </label>
      </div>
    </section>

    <section class="transcript">
      <div class="transcript-header">
        <h2>Transcript</h2>
        <button class="clear" on:click={() => (transcriptLines = [])}>Clear</button>
      </div>
      <pre>{transcriptLines.join("\n")}</pre>
    </section>
  </main>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: "Segoe UI", sans-serif;
    background:
      radial-gradient(circle at top left, rgba(245, 182, 84, 0.28), transparent 32%),
      linear-gradient(180deg, #f6efe1 0%, #ebf1f5 100%);
    color: #1e2330;
  }

  .shell {
    min-height: 100vh;
    padding: 24px;
  }

  .card {
    max-width: 1100px;
    margin: 0 auto;
    padding: 28px;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 20px 60px rgba(26, 38, 54, 0.12);
    backdrop-filter: blur(12px);
  }

  .hero {
    margin-bottom: 24px;
  }

  .eyebrow {
    margin: 0 0 8px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.76rem;
    color: #8f5b1b;
    font-weight: 700;
  }

  h1 {
    margin: 0;
    font-size: clamp(2rem, 5vw, 3.5rem);
    line-height: 1.05;
  }

  .lede {
    max-width: 760px;
    color: #475467;
    font-size: 1.05rem;
  }

  .controls {
    display: grid;
    gap: 18px;
    margin-bottom: 24px;
  }

  .button-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px 16px;
  }

  label {
    display: grid;
    gap: 6px;
    font-weight: 600;
  }

  input[type="number"],
  select {
    border: 1px solid #cdd5df;
    border-radius: 12px;
    padding: 10px 12px;
    font: inherit;
    background: #fff;
  }

  button {
    border: 0;
    border-radius: 999px;
    padding: 12px 18px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
  }

  .start {
    background: #1f7a46;
    color: #fff;
  }

  .stop,
  .clear {
    background: #a63232;
    color: #fff;
  }

  .status {
    font-weight: 700;
    color: #475467;
  }

  .transcript {
    border-radius: 20px;
    background: #111827;
    color: #e5eef8;
    padding: 18px;
  }

  .transcript-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .transcript-header h2 {
    margin: 0;
    font-size: 1rem;
  }

  pre {
    margin: 0;
    white-space: pre-wrap;
    min-height: 280px;
    max-height: 52vh;
    overflow: auto;
    font-family: "Cascadia Code", "SFMono-Regular", monospace;
    font-size: 0.95rem;
    line-height: 1.45;
  }

  @media (max-width: 720px) {
    .shell {
      padding: 12px;
    }

    .card {
      padding: 18px;
    }

    .button-row {
      align-items: stretch;
    }

    button {
      width: 100%;
    }
  }
</style>
