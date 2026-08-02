import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const source = join(root, 'dist', 'index.html')
const target = join(root, '..', 'src', 'zephyr', 'dashboard.html')
const html = readFileSync(source, 'utf8')
const external = html.match(/(?:src|href)="[^"]+\.(?:js|css|woff2?|png|svg)"/g)
const output = html.endsWith('\n') ? html : `${html}\n`

if (external) {
  throw new Error(`dist/index.html references external assets: ${external.join(', ')}`)
}
if (!html.includes('<base href="/"')) {
  throw new Error('dist/index.html does not contain the proxy base element')
}
if (process.argv.includes('--check')) {
  if (readFileSync(target, 'utf8') !== output) {
    throw new Error('dashboard.html is stale; run npm run build and commit the result')
  }
  console.log('dashboard.html matches the frontend source')
  process.exit(0)
}
writeFileSync(target, output)
console.log(`wrote ${target} (${(html.length / 1024).toFixed(0)} KiB)`)
