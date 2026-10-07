// The subset of Markdown the apps display, turned into a tree.
//
// A parser rather than a dependency, and a tree rather than a string of HTML.
// The text comes from wiki notes, dataset files and model output, so it is
// attacker-influenced, and handing it to `v-html` would put an injection point
// on the page. A tree renders through `{{ }}`, where markup is text.
//
// Covered: headings, paragraphs, fenced code, bullet and numbered lists, block
// quotes, and inline code, bold, italic and links. Tables, footnotes and
// reference links are not, and their source stays on the screen as the
// characters the author typed, which is the failure this is allowed to have.

/**
 * A run of text inside a paragraph, heading or list item.
 *
 * Emphasis is a flag rather than a kind, so the list stays flat. `**a `b`**` is
 * a bold word and a bold code word, not a bold thing containing a code thing,
 * and a flat list is one a template draws in one loop instead of through a
 * component that recurses into itself.
 */
export type Span = {
  kind: 'text' | 'code' | 'link'
  text: string
  href?: string
  strong?: boolean
  emphasis?: boolean
}

export type Block =
  | { kind: 'heading'; level: number; spans: Span[] }
  | { kind: 'paragraph'; spans: Span[] }
  | { kind: 'quote'; spans: Span[] }
  | { kind: 'code'; language: string; text: string }
  | { kind: 'list'; ordered: boolean; items: Span[][] }
  | { kind: 'rule' }

const HEADING = /^(#{1,6})\s+(.*)$/
const BULLET = /^\s{0,3}[-*+]\s+(.*)$/
const NUMBER = /^\s{0,3}\d+[.)]\s+(.*)$/
const QUOTE = /^\s{0,3}>\s?(.*)$/
const FENCE = /^\s{0,3}(`{3,}|~{3,})\s*(\S*)/
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/

/**
 * The blocks of one message.
 *
 * Line-driven, because the shapes above are line shapes and a character-level
 * scanner would be a larger thing to be wrong about.
 */
export function blocks(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, '\n').split('\n')
  const out: Block[] = []
  let at = 0

  // A run of lines that all match `test`, with `take` pulling the content out
  // of each. Used for lists and quotes, which are one block over many lines.
  const run = (test: RegExp, take: (line: string) => string): string[] => {
    const taken: string[] = []
    while (at < lines.length && test.test(lines[at])) {
      taken.push(take(lines[at]))
      at += 1
    }
    return taken
  }

  while (at < lines.length) {
    const line = lines[at]

    if (!line.trim()) {
      at += 1
      continue
    }

    const fence = FENCE.exec(line)
    if (fence) {
      const closing = fence[1][0]
      at += 1
      const body: string[] = []
      // An unterminated fence runs to the end, which is what a transcript
      // streaming a code block one chunk at a time looks like mid-turn.
      while (at < lines.length && !new RegExp(`^\\s{0,3}${closing}{3,}\\s*$`).test(lines[at])) {
        body.push(lines[at])
        at += 1
      }
      at += 1
      out.push({ kind: 'code', language: fence[2] ?? '', text: body.join('\n') })
      continue
    }

    if (RULE.test(line)) {
      out.push({ kind: 'rule' })
      at += 1
      continue
    }

    const heading = HEADING.exec(line)
    if (heading) {
      out.push({ kind: 'heading', level: heading[1].length, spans: spans(heading[2]) })
      at += 1
      continue
    }

    if (BULLET.test(line) || NUMBER.test(line)) {
      const ordered = !BULLET.test(line)
      const pattern = ordered ? NUMBER : BULLET
      const items = run(pattern, (each) => pattern.exec(each)![1])
      out.push({ kind: 'list', ordered, items: items.map(spans) })
      continue
    }

    if (QUOTE.test(line)) {
      const said = run(QUOTE, (each) => QUOTE.exec(each)![1])
      out.push({ kind: 'quote', spans: spans(said.join(' ')) })
      continue
    }

    // A paragraph is everything up to a blank line or the start of another
    // block, joined with spaces: a hard-wrapped paragraph is one paragraph.
    const said: string[] = []
    while (at < lines.length && lines[at].trim()) {
      const next = lines[at]
      if (
        HEADING.test(next) ||
        BULLET.test(next) ||
        NUMBER.test(next) ||
        QUOTE.test(next) ||
        FENCE.test(next) ||
        RULE.test(next)
      ) {
        break
      }
      said.push(next.trim())
      at += 1
    }
    out.push({ kind: 'paragraph', spans: spans(said.join(' ')) })
  }

  return out
}

// Code first, so a backtick span containing `**` or `_` keeps them. Bold before
// italic, because `**x**` also matches the italic pattern.
const INLINE =
  /(`+)([^`]|[^`].*?[^`])\1|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*]+)\*|(?<![\w\\])_([^_]+)_(?!\w)|\[([^\]]*)\]\(([^)\s]+)[^)]*\)/s

/** The spans of one line of text. */
export function spans(source: string): Span[] {
  const out: Span[] = []
  let rest = source
  for (;;) {
    const found = INLINE.exec(rest)
    if (!found) break
    if (found.index > 0) out.push({ kind: 'text', text: rest.slice(0, found.index) })

    const strong = found[3] ?? found[4]
    const emphasis = found[5] ?? found[6]
    if (found[2] !== undefined) out.push({ kind: 'code', text: found[2].trim() })
    else if (strong !== undefined) out.push(...mark(spans(strong), { strong: true }))
    else if (emphasis !== undefined) out.push(...mark(spans(emphasis), { emphasis: true }))
    else out.push({ kind: 'link', text: found[7] || found[8], href: found[8] })

    rest = rest.slice(found.index + found[0].length)
  }
  if (rest) out.push({ kind: 'text', text: rest })
  return out
}

/** Set a flag on every span of a run, keeping the flags already on it. */
function mark(run: Span[], flag: { strong?: true; emphasis?: true }): Span[] {
  return run.map((span) => ({ ...span, ...flag }))
}

/**
 * Whether a link may be followed.
 *
 * The href comes out of a model's message, so `javascript:` is a real thing to
 * refuse. Anything not plainly http, https or a fragment is drawn as text.
 */
export function followable(href: string): boolean {
  return /^(https?:\/\/|\/|#)/i.test(href)
}
