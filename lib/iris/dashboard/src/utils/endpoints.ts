// Controller endpoint-proxy helpers.
//
// A registered endpoint named `/tunix/inference/server` is reachable through
// the controller's reverse proxy at `/proxy/tunix.inference.server`: the proxy
// decodes the path component by replacing '.' with '/' and resolving both the
// slash-prefixed and bare forms. The encoding has no escape for a literal '.',
// so a name containing one cannot be turned into a working proxy link.

/** Whether `name` can be encoded into a working proxy path (no literal '.'). */
export function canProxyEndpoint(name: string): boolean {
  return name.length > 0 && !name.includes('.')
}

/**
 * Proxy path for an endpoint name, e.g. `/tunix/inference/server` ->
 * `/proxy/tunix.inference.server`. The leading slash is dropped and remaining
 * slashes become dots to match the proxy's decoding. Gate on
 * {@link canProxyEndpoint} before rendering the result as a link.
 */
export function proxyPathForEndpoint(name: string): string {
  const trimmed = name.replace(/^\/+/, '')
  return `/proxy/${trimmed.split('/').join('.')}`
}

/** Short label for an endpoint: its last path segment (or the whole name). */
export function endpointLabel(name: string): string {
  const trimmed = name.replace(/\/+$/, '')
  const slash = trimmed.lastIndexOf('/')
  return slash >= 0 ? trimmed.slice(slash + 1) : trimmed
}
