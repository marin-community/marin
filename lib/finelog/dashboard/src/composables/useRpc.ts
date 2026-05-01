/**
 * Connect-RPC composable for finelog services.
 *
 * Wraps fetch() with reactive loading/error state. Caller invokes refresh() to fetch.
 * Body may be a static record or a factory that closes over reactive state.
 */
import { ref, type Ref } from 'vue'

export type RpcBody = Record<string, unknown> | (() => Record<string, unknown>)

export interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

function useRpc<T>(service: string, method: string, body?: RpcBody): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)
  let generation = 0

  async function refresh() {
    const gen = ++generation
    loading.value = true
    error.value = null
    try {
      const resolvedBody = typeof body === 'function' ? body() : (body ?? {})
      const resp = await fetch(`${service}/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(resolvedBody),
      })
      if (gen !== generation) return
      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`${method}: ${resp.status} ${resp.statusText}${text ? ` — ${text}` : ''}`)
      }
      const payload = (await resp.json()) as T
      if (gen !== generation) return
      data.value = payload
    } catch (e) {
      if (gen !== generation) return
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (gen === generation) loading.value = false
    }
  }

  return { data, loading, error, refresh }
}

export function useLogServiceRpc<T>(method: string, body?: RpcBody): RpcState<T> {
  return useRpc<T>('finelog.logging.LogService', method, body)
}

export function useStatsRpc<T>(method: string, body?: RpcBody): RpcState<T> {
  return useRpc<T>('finelog.stats.StatsService', method, body)
}

export async function statsRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  const resp = await fetch(`finelog.stats.StatsService/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${method}: ${resp.status} ${resp.statusText}${text ? ` — ${text}` : ''}`)
  }
  return resp.json() as Promise<T>
}

export async function logRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  const resp = await fetch(`finelog.logging.LogService/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`${method}: ${resp.status} ${resp.statusText}${text ? ` — ${text}` : ''}`)
  }
  return resp.json() as Promise<T>
}
