import type { ServingInfo } from './types'

/** Resolve a path relative to the page URL. The dashboard is served under the
 * Iris controller proxy at /proxy/<name>/, and the proxy does not rewrite
 * bodies — an absolute path like /v1/chat/completions would escape the prefix. */
export const api = (path: string) => new URL(path, location.href).toString()

export async function fetchInfo(): Promise<ServingInfo> {
  const response = await fetch(api('info'))
  if (!response.ok) throw new Error(`info returned ${response.status}`)
  return response.json()
}

export interface HealthResult {
  ok: boolean
  model: string | null
}

export async function fetchHealth(): Promise<HealthResult> {
  const response = await fetch(api('health'))
  const body = await response.json().catch(() => ({}))
  return { ok: response.ok, model: body.model ?? null }
}

/** POST an OpenAI request and invoke onData for either buffered JSON or SSE events. */
export async function requestCompletion(
  path: string,
  body: Record<string, unknown>,
  streaming: boolean,
  signal: AbortSignal,
  onData: (data: any) => void,
): Promise<void> {
  const response = await fetch(api(path), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`${response.status} — ${await response.text()}`)
  }
  if (!streaming) {
    onData(await response.json())
    return
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue
      const payload = trimmed.slice(5).trim()
      if (payload === '[DONE]') return
      try {
        onData(JSON.parse(payload))
      } catch {
        // Skip keepalives and partial frames.
      }
    }
  }
}
