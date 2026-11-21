// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Serves the public Marin Belay page at marin.community/belay/ with a clean address bar. The page is a public Marina
// applet, which Marina only serves by id (public.applets.marina.oa.dev/a/<id>/), so this Cloudflare Worker, routed on
// marin.community/belay*, fetches each request from there and returns it under /belay/. The page and its API calls
// use paths relative to the address bar, so only Marina's redirects (the root to its versioned path) need rewriting.
// Read-only: anything but GET or HEAD is refused, as Marina's public host refuses it.

const ORIGIN = 'https://public.applets.marina.oa.dev'
const APP = '/a/33801d46-e280-46f9-b994-2c78603cdf15'
const PREFIX = '/belay'
// Request headers worth passing on: content negotiation and conditional requests. Cookies and the rest stay behind.
const FORWARDED = ['accept', 'accept-encoding', 'if-none-match', 'if-modified-since']

export function upstreamUrl(url) {
  return new URL(APP + url.pathname.slice(PREFIX.length) + url.search, ORIGIN)
}

// A Marina redirect into the app, moved under /belay/; any other location is left as it is.
export function localLocation(location, upstream) {
  const target = new URL(location, upstream)
  if (target.origin !== ORIGIN || !(target.pathname === APP || target.pathname.startsWith(`${APP}/`))) return location
  return PREFIX + (target.pathname.slice(APP.length) || '/') + target.search + target.hash
}

export default {
  async fetch(request) {
    const url = new URL(request.url)
    if (url.pathname === PREFIX) return Response.redirect(`${url.origin}${PREFIX}/${url.search}`, 308)
    if (!url.pathname.startsWith(`${PREFIX}/`)) return fetch(request)
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return new Response('method not allowed', { status: 405, headers: { allow: 'GET, HEAD' } })
    }
    const upstream = upstreamUrl(url)
    const headers = new Headers()
    for (const name of FORWARDED) {
      const value = request.headers.get(name)
      if (value) headers.set(name, value)
    }
    const response = await fetch(upstream, { method: request.method, headers, redirect: 'manual' })
    const out = new Headers(response.headers)
    const location = out.get('location')
    if (location) out.set('location', localLocation(location, upstream))
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers: out })
  },
}
