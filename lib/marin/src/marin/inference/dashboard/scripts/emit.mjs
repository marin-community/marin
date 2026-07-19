// Copy the built single-file dashboard into the marin package, where
// quick_serve_dashboard.py serves it via importlib.resources. Fails if the
// build emitted any external asset reference — the artifact must be fully
// self-contained (inlineScripts/inlineStyles in rsbuild.config.ts).
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const source = join(root, 'dist', 'index.html')
const target = join(root, '..', 'quick_serve_dashboard.html')

const html = readFileSync(source, 'utf8')
const external = html.match(/(?:src|href)="[^"]+\.(?:js|css|woff2?|png|svg)"/g)
if (external) {
  throw new Error(`dist/index.html references external assets (inlining broke): ${external.join(', ')}`)
}
writeFileSync(target, html.endsWith('\n') ? html : html + '\n')
console.log(`wrote ${target} (${(html.length / 1024).toFixed(0)} KiB)`)
