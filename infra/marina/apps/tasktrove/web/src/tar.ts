// A tar archive, read out of its bytes.
//
// A TaskTrove task is a gzipped tar, and this is the half that reads the tar.
// ustar headers are 512-byte blocks with the name, the size in octal and a
// type flag; GNU long names arrive as an `L` entry whose body is the name of
// the entry after it. Directories are kept so the tree can show them empty.

export type Entry = {
  path: string
  size: number
  directory: boolean
  bytes: Uint8Array
}

const BLOCK = 512
const ascii = new TextDecoder('ascii')
const utf8 = new TextDecoder('utf-8', { fatal: true })

function field(block: Uint8Array, start: number, length: number): string {
  const raw = block.subarray(start, start + length)
  const end = raw.indexOf(0)
  return ascii.decode(end < 0 ? raw : raw.subarray(0, end))
}

function octal(block: Uint8Array, start: number, length: number): number {
  const text = field(block, start, length).trim()
  return text ? parseInt(text, 8) : 0
}

/** Every entry in the archive, in order. */
export function entries(archive: Uint8Array): Entry[] {
  const found: Entry[] = []
  let at = 0
  let longName: string | undefined
  while (at + BLOCK <= archive.length) {
    const block = archive.subarray(at, at + BLOCK)
    if (block.every((byte) => byte === 0)) break
    const size = octal(block, 124, 12)
    const flag = String.fromCharCode(block[156] ?? 0)
    const prefix = field(block, 345, 155)
    const name = longName ?? (prefix ? `${prefix}/${field(block, 0, 100)}` : field(block, 0, 100))
    longName = undefined
    const body = archive.subarray(at + BLOCK, at + BLOCK + size)
    at += BLOCK + Math.ceil(size / BLOCK) * BLOCK
    if (flag === 'L') {
      longName = field(body, 0, size)
      continue
    }
    if (flag !== '0' && flag !== '\0' && flag !== '5') continue
    found.push({
      path: name.replace(/^\.\//, '').replace(/\/$/, ''),
      size,
      directory: flag === '5',
      bytes: body,
    })
  }
  return found
}

/** The bytes as text, or nothing when they are not UTF-8. */
export function text(bytes: Uint8Array): string | undefined {
  try {
    return utf8.decode(bytes)
  } catch {
    return undefined
  }
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

export function decodeBase64(encoded: string): Uint8Array {
  const binary = atob(encoded)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}
