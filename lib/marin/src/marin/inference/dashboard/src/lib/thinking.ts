const OPEN = '<think>'
const CLOSE = '</think>'

export interface ThinkingSplit {
  thinking: string
  visible: string
  /** True while an opened thinking segment has not been closed yet. */
  inThinking: boolean
}

/** Longest suffix of `text` that is a strict prefix of an unfinished tag, so a
 * streaming cut mid-`</think>` is neither shown nor treated as content. */
function trailingPartialTag(text: string): string {
  const max = Math.min(text.length, CLOSE.length - 1)
  for (let len = max; len > 0; len--) {
    const suffix = text.slice(text.length - len)
    if (OPEN.startsWith(suffix) || CLOSE.startsWith(suffix)) return suffix
  }
  return ''
}

/** Split accumulated model output into a thinking segment and visible text.
 *
 * Handles models that emit raw tags in content (Qwen3-style `<think>…</think>`
 * at the start of the reply) and templates that open the block inside the
 * prompt, so the output carries only a bare closing tag. Re-run on the full
 * accumulated text after each streaming delta.
 */
export function splitThinking(raw: string): ThinkingSplit {
  const held = trailingPartialTag(raw)
  const text = held ? raw.slice(0, raw.length - held.length) : raw

  const lead = text.length - text.trimStart().length
  const openIdx = text.startsWith(OPEN, lead) ? lead : -1
  const closeIdx = text.indexOf(CLOSE)

  if (closeIdx !== -1 && (openIdx === -1 || closeIdx < openIdx)) {
    // Bare closing tag: everything before it was reasoning.
    return {
      thinking: text.slice(0, closeIdx).trim(),
      visible: text.slice(closeIdx + CLOSE.length).replace(/^\s+/, ''),
      inThinking: false,
    }
  }
  if (openIdx !== -1) {
    const after = text.slice(openIdx + OPEN.length)
    const end = after.indexOf(CLOSE)
    if (end === -1) {
      return { thinking: after.replace(/^\s+/, ''), visible: '', inThinking: true }
    }
    return {
      thinking: after.slice(0, end).trim(),
      visible: after.slice(end + CLOSE.length).replace(/^\s+/, ''),
      inThinking: false,
    }
  }
  return { thinking: '', visible: text, inThinking: false }
}
