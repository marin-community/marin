/**
 * Open captured profiles in the bundled speedscope viewer.
 *
 * Speedscope's prebuilt SPA is served same-origin at `/static/speedscope/`
 * (copied from its npm package at build time). We hand it the profile bytes via
 * a blob URL referenced through speedscope's `profileURL` hash param; the blob
 * is same-origin with the viewer, so the fetch succeeds — the original
 * blob-URL-to-speedscope.app approach failed because cross-origin pages cannot
 * read `blob:` URLs.
 *
 * Capturing a profile takes ~10s, long enough that a `window.open` issued after
 * the RPC resolves is suppressed by popup blockers (the open is no longer tied
 * to the click gesture). So callers open the window synchronously inside the
 * click handler via `openSpeedscopeWindow()` and call `show()` once the bytes
 * arrive.
 */

// Point at index.html explicitly: the controller serves static assets via
// Starlette StaticFiles without directory-index fallback, so a bare
// `/static/speedscope/` 404s. Speedscope's own assets load by relative path, so
// they still resolve under /static/speedscope/.
const SPEEDSCOPE_URL = '/static/speedscope/index.html'

// Keep the blob URL alive long enough for the viewer window to fetch it, then
// revoke so the bytes don't leak for the lifetime of the dashboard tab.
const BLOB_TTL_MS = 60_000

const LOADING_HTML =
  '<!doctype html><meta charset="utf-8"><title>Loading profile…</title>' +
  '<body style="margin:0;font:14px system-ui,sans-serif;color:#888;display:flex;' +
  'height:100vh;align-items:center;justify-content:center">Capturing profile…</body>'

export interface PendingSpeedscope {
  /** Load `bytes` (a speedscope-format profile) into the opened viewer window. */
  show(bytes: Uint8Array, title: string): void
  /** Abandon the viewer window (the profile RPC failed or returned nothing). */
  cancel(): void
}

/**
 * Open a viewer window now — inside the user gesture — and return a handle to
 * fill it once the profile bytes are available. Opening eagerly keeps the
 * window from being blocked as a popup after the profiling RPC's latency.
 */
export function openSpeedscopeWindow(): PendingSpeedscope {
  const win = window.open('', '_blank')
  if (win) {
    win.document.write(LOADING_HTML)
    win.document.close()
  }
  return {
    show(bytes: Uint8Array, title: string) {
      const url = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: 'application/json' }))
      const target = `${SPEEDSCOPE_URL}#profileURL=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`
      // Navigate the already-open window (path changes from about:blank, so
      // speedscope loads fresh and reads the hash param on mount). Fall back to
      // a fresh open if the eager open was blocked.
      if (win) win.location.href = target
      else window.open(target, '_blank')
      setTimeout(() => URL.revokeObjectURL(url), BLOB_TTL_MS)
    },
    cancel() {
      win?.close()
    },
  }
}
