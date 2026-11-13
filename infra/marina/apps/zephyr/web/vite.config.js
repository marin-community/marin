import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";

export default defineConfig({
  base: "./",
  plugins: [vue()],
  resolve: {
    alias: {
      vue: fileURLToPath(new URL("./node_modules/vue", import.meta.url)),
      "@marina": fileURLToPath(new URL("../../../web", import.meta.url)),
    },
  },
  build: { outDir: "../dist", emptyOutDir: true },
});
