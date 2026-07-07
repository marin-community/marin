import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite serves the React app from web/ during dev (HMR) and bundles to
// web/dist for production. The Hono server (server/main.ts) serves
// web/dist as static assets in prod, and proxies /api/* to the server
// during dev.
export default defineConfig({
  root: path.resolve(__dirname, "web"),
  build: {
    outDir: path.resolve(__dirname, "web/dist"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // API_PROXY lets dev environments where :8080 is taken point the
      // proxy at a server started with a custom PORT.
      "/api": process.env.API_PROXY ?? "http://localhost:8080",
    },
  },
  plugins: [react()],
});
