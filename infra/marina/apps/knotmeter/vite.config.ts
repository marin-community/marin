import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
 base: "./",
 plugins: [tailwindcss()],
 resolve: { alias: { "@": new URL("./src/upstream", import.meta.url).pathname, "@marina": new URL("../../web", import.meta.url).pathname } },
});
