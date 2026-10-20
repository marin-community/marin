import { build } from "esbuild";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const shared = fileURLToPath(new URL(".", import.meta.url));
const app = process.cwd();
const dist = resolve(app, "dist");
await mkdir(dist, { recursive: true });
await build({
  entryPoints: [resolve(shared, "main.js")],
  outfile: resolve(dist, "main.js"),
  bundle: true,
  format: "iife",
});
const template = await readFile(resolve(app, "index.html"), "utf8");
const content = await readFile(resolve(shared, "content.html"), "utf8");
await writeFile(resolve(dist, "index.html"), template.replace("    <!-- CONVERSATION_CONTENT -->\n", content));
await copyFile(resolve(shared, "markdown.css"), resolve(dist, "markdown.css"));
const style = await readFile(resolve(shared, "style.css"), "utf8");
const extra = await readFile(resolve(app, "style.css"), "utf8");
await writeFile(resolve(dist, "style.css"), style + extra);
