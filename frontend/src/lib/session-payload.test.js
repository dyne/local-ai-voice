import { describe, expect, it } from "vitest";

import { buildSessionPayload, preferredMimeType, wsBaseUrl } from "./session-payload";

describe("preferredMimeType", () => {
  it("returns the first supported opus type", () => {
    const fakeRecorder = {
      isTypeSupported(value) {
        return value === "audio/webm;codecs=opus";
      },
    };

    expect(preferredMimeType(fakeRecorder)).toBe("audio/webm;codecs=opus");
  });

  it("returns empty string when nothing is supported", () => {
    const fakeRecorder = {
      isTypeSupported() {
        return false;
      },
    };

    expect(preferredMimeType(fakeRecorder)).toBe("");
  });
});

describe("buildSessionPayload", () => {
  it("normalizes browser form state into the backend payload", () => {
    expect(
      buildSessionPayload({
        sessionId: "abc",
        saveSample: true,
        silenceDetect: false,
        debug: true,
        vadMode: "2",
        chunkSeconds: "1.5",
        overlapSeconds: "0.25",
        mimeType: "audio/webm;codecs=opus",
      }),
    ).toEqual({
      session_id: "abc",
      save_sample: true,
      silence_detect: false,
      debug: true,
      vad_mode: 2,
      chunk_seconds: 1.5,
      overlap_seconds: 0.25,
      mime_type: "audio/webm;codecs=opus",
      audio_bitrate: 48000,
    });
  });
});

describe("wsBaseUrl", () => {
  it("uses ws for http origins", () => {
    expect(wsBaseUrl({ protocol: "http:", host: "127.0.0.1:8000" })).toBe(
      "ws://127.0.0.1:8000",
    );
  });

  it("uses wss for https origins", () => {
    expect(wsBaseUrl({ protocol: "https:", host: "example.com" })).toBe("wss://example.com");
  });
});
