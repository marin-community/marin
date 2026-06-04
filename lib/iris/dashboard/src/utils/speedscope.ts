/**
 * Open a captured profile in the bundled speedscope viewer.
 *
 * Speedscope's prebuilt SPA is served from the same origin at
 * `/static/speedscope/` (copied from its npm package at build time). We hand it
 * the profile bytes via a blob URL referenced through speedscope's `profileURL`
 * hash param. The blob is same-origin with the viewer, so the fetch succeeds —
 * the original blob-URL-to-speedscope.app approach failed because cross-origin
 * pages cannot read `blob:` URLs.
 */

const SPEEDSCOPE_URL = '/static/speedscope/'

// Keep the blob URL alive long enough for the newly opened tab to fetch it,
// then revoke so the bytes don't leak for the lifetime of the dashboard tab.
const BLOB_TTL_MS = 60_000

/** Open `bytes` (a speedscope-format profile) in the bundled viewer in a new tab. */
export function openInSpeedscope(bytes: Uint8Array, title: string): void {
  const blob = new Blob([bytes], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const hash = `profileURL=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`
  window.open(`${SPEEDSCOPE_URL}#${hash}`, '_blank')
  setTimeout(() => URL.revokeObjectURL(url), BLOB_TTL_MS)
}
