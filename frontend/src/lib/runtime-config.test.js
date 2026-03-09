import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG, readRuntimeConfig } from "./runtime-config";

describe("readRuntimeConfig", () => {
  it("returns defaults when no runtime config exists", () => {
    expect(readRuntimeConfig({})).toEqual(DEFAULT_CONFIG);
  });

  it("uses injected runtime config values when valid", () => {
    expect(
      readRuntimeConfig({
        __LOCAL_AI_CONFIG__: {
          silenceDetectDefault: false,
          vadModeDefault: 1,
        },
      }),
    ).toEqual({
      silenceDetectDefault: false,
      vadModeDefault: 1,
    });
  });

  it("falls back for invalid values", () => {
    expect(
      readRuntimeConfig({
        __LOCAL_AI_CONFIG__: {
          silenceDetectDefault: "yes",
          vadModeDefault: 99,
        },
      }),
    ).toEqual(DEFAULT_CONFIG);
  });
});
