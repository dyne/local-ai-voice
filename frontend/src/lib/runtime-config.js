const DEFAULT_CONFIG = Object.freeze({
  silenceDetectDefault: true,
  vadModeDefault: 3,
});

export function readRuntimeConfig(globalWindow = window) {
  const value = globalWindow?.__LOCAL_AI_CONFIG__;
  if (!value || typeof value !== "object") {
    return { ...DEFAULT_CONFIG };
  }
  return {
    silenceDetectDefault:
      typeof value.silenceDetectDefault === "boolean"
        ? value.silenceDetectDefault
        : DEFAULT_CONFIG.silenceDetectDefault,
    vadModeDefault:
      Number.isInteger(value.vadModeDefault) && value.vadModeDefault >= 0 && value.vadModeDefault <= 3
        ? value.vadModeDefault
        : DEFAULT_CONFIG.vadModeDefault,
  };
}

export { DEFAULT_CONFIG };
