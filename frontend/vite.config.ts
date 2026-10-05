import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  const target = env.BACKEND_PROXY_URL || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 3000,
      strictPort: true,
      proxy: Object.fromEntries(
        [
          "/health",
          "/status",
          "/scene",
          "/detections",
          "/metrics",
          "/api",
          "/ws",
        ].map((path) => [path, { target, ws: path === "/ws" }]),
      ),
    },
  };
});
