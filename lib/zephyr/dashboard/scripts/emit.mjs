import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const source = join(root, 'dist', 'index.html')
const target = join(root, '..', 'src', 'zephyr', 'dashboard.html')
const html = readFileSync(source, 'utf8')
const external = html.match(/(?:src|href)="[^"]+\.(?:js|css|woff2?|png|svg)"/g)

function filesUnder(path) {
  if (!statSync(path).isDirectory()) return [path]
  return readdirSync(path).flatMap((name) => filesUnder(join(path, name)))
}

const sourceFiles = [
  ...filesUnder(join(root, 'src')),
  ...filesUnder(join(root, 'scripts')),
  ...[
    'env.d.ts',
    'package-lock.json',
    'package.json',
    'postcss.config.cjs',
    'rsbuild.config.ts',
    'tailwind.config.ts',
    'tsconfig.json',
  ].map((name) => join(root, name)),
].sort((left, right) => relative(root, left).localeCompare(relative(root, right)))
const hash = createHash('sha256')
for (const path of sourceFiles) {
  hash.update(relative(root, path).replaceAll('\\', '/'))
  hash.update('\0')
  hash.update(readFileSync(path))
  hash.update('\0')
}
const sourceHash = hash.digest('hex')
const stampedHtml = html.replace('<html ', `<html data-source-hash="${sourceHash}" `)
const output = stampedHtml.endsWith('\n') ? stampedHtml : `${stampedHtml}\n`

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
