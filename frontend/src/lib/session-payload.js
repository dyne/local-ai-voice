export const TARGET_AUDIO_BITRATE = 48000;
export const CHUNK_TIMESLICE_MS = 250;

export function preferredMimeType(mediaRecorderApi = globalThis.MediaRecorder) {
  const candidates = [
    "audio/ogg;codecs=opus",
    "audio/webm;codecs=opus",
    "audio/ogg",
    "audio/webm",
  ];
  if (!mediaRecorderApi || typeof mediaRecorderApi.isTypeSupported !== "function") {
    return "";
  }
  for (const candidate of candidates) {
    if (mediaRecorderApi.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return "";
}

export function buildSessionPayload({
  sessionId,
  saveSample,
  silenceDetect,
  debug,
  vadMode,
  chunkSeconds,
  overlapSeconds,
  mimeType,
  audioBitrate = TARGET_AUDIO_BITRATE,
}) {
  return {
    session_id: sessionId,
    save_sample: Boolean(saveSample),
    silence_detect: Boolean(silenceDetect),
    debug: Boolean(debug),
    vad_mode: Number(vadMode),
    chunk_seconds: Number(chunkSeconds),
    overlap_seconds: Number(overlapSeconds),
    mime_type: mimeType,
    audio_bitrate: Number(audioBitrate),
  };
}

export function wsBaseUrl(locationObject = window.location) {
  const proto = locationObject.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${locationObject.host}`;
}
