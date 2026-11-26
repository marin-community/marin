import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
export default defineConfig({
  base: "./",
  plugins: [vue()],
  resolve: {
    alias: {
      vue: "vue/dist/vue.runtime.esm-bundler.js",
      "@marina": new URL("../../web", import.meta.url).pathname,
    },
  },
});
