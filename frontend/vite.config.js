import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/session": "http://127.0.0.1:8000",
      "/events": "http://127.0.0.1:8000",
      "/audio": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.js"],
  },
});
