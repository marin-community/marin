import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    outDir: fileURLToPath(
      new URL("../reference_outputs/mixture_fit_observatory_20260713", import.meta.url),
    ),
    emptyOutDir: true,
    target: "es2022",
  },
});
