// A URN names one row in one app.
//
// `marin:{app}:{kind}:{id}`. Every app on this origin is served under `/{app}/`
// and shows a row of `{kind}` at `/{app}/{kind}/{id}`, so a URN is an address
// this file can build without asking anything: no registry, no lookup, no
// client. An app that wants its rows linkable gives them that route.

/** One `{app}` or `{kind}` segment, whole. Lowercase, starts with a letter. */
const SEGMENT = /^[a-z][a-z0-9_-]{0,62}$/

/** One `{id}` segment, whole. A URL-safe token: no colon, no slash, no space. */
const ID = /^[A-Za-z0-9][A-Za-z0-9._~-]{0,199}$/

/**
 * A URN anywhere in a run of text.
 *
 * Four colon-separated parts is the whole discriminator, so a `marin:` prefix
 * followed by fewer parts — a namespace someone wrote in prose — does not
 * match. An id is matched as written; `urn()` is what normalizes one.
 */
export const URN = /marin:[a-z][a-z0-9_-]{0,62}:[a-z][a-z0-9_-]{0,62}:[A-Za-z0-9][A-Za-z0-9._~-]{0,199}/g

/** The parts of a URN. */
export interface Parts {
  app: string
  kind: string
  id: string
}

/**
 * Mint `marin:{app}:{kind}:{id}` — the one constructor, so a page never
 * assembles a URN a strict parser would refuse.
 *
 * Throws on a part the grammar rejects; a caller passes its own app and kind as
 * literals and a row id the app answered with.
 */
export function urn(app: string, kind: string, id: string): string {
  if (!SEGMENT.test(app)) throw new Error(`\`${app}\` is not a URN app segment`)
  if (!SEGMENT.test(kind)) throw new Error(`\`${kind}\` is not a URN kind segment`)
  if (!ID.test(id)) throw new Error(`\`${id}\` is not a URN id`)
  return `marin:${app}:${kind}:${id}`
}

/** Whether the whole of this text is one URN. */
export function isUrn(text: string): boolean {
  return new RegExp(`^${URN.source}$`).test(text.trim())
}

/** The parts of one URN, or nothing when the text is not one. */
export function parse(text: string): Parts | undefined {
  if (!isUrn(text)) return undefined
  const [, app, kind, id] = text.trim().split(':')
  return { app, kind, id }
}

/**
 * What a person pasted, as a URN.
 *
 * A URN arrives quoted, backticked, wrapped in angle brackets or with a comma
 * still on the end, because it was copied out of a transcript or a log line.
 * Take the first one in the text rather than refusing the whole paste.
 */
export function typed(text: string): string {
  return text.match(new RegExp(URN.source))?.[0] ?? text.trim()
}

/**
 * The path that shows one URN's row: `/{app}/{kind}/{id}`.
 *
 * An origin-relative path rather than a router path, because the app that holds
 * the row is usually not the app rendering the link, and the two are separate
 * bundles behind separate prefixes.
 */
export function path(text: string): string | undefined {
  const parts = parse(text)
  if (!parts) return undefined
  return `/${parts.app}/${parts.kind}/${encodeURIComponent(parts.id)}`
}

/** One piece of a run of text. */
export interface Mention {
  /** The prose as it was written, or what the link says. */
  text: string
  /** The row this piece names. Empty for prose. */
  urn: string
  /** Whether `text` is the writer's own words for the link. */
  named: boolean
}

/**
 * A writer naming a row: `[the sampled task](marin:tasktrove:task:41207)`.
 *
 * The shape is markdown's, and the target is a URN rather than an address. A
 * model writing prose reaches for these brackets without being taught them, and
 * the words in them are what the link says.
 */
const NAMED = new RegExp(`\\[([^\\]\\n]{1,200})\\]\\((${URN.source})\\)|(${URN.source})`, 'g')

/**
 * A run of text, split where it names a row.
 *
 * Two shapes are a link: a URN in brackets after the words for it, and a bare
 * URN, which says itself. Everything else comes back as it was written, so
 * rendering the pieces in order renders the text.
 */
export function mentions(text: string): Mention[] {
  const pieces: Mention[] = []
  let at = 0
  for (const found of text.matchAll(NAMED)) {
    const start = found.index ?? 0
    if (start > at) pieces.push({ text: text.slice(at, start), urn: '', named: false })
    if (found[2]) pieces.push({ text: found[1], urn: found[2], named: true })
    else pieces.push({ text: found[3], urn: found[3], named: false })
    at = start + found[0].length
  }
  if (at < text.length) pieces.push({ text: text.slice(at), urn: '', named: false })
  return pieces
}
