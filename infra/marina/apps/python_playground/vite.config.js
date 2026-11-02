import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [vue()],
  resolve: {
    alias: {
      vue: new URL("./node_modules/vue", import.meta.url).pathname,
      "@marina": new URL("../../web", import.meta.url).pathname,
    },
  },
});
