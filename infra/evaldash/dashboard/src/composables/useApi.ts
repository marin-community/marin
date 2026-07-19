/**
 * Minimal reactive GET wrapper over fetch().
 *
 * `path` is a factory so it can close over reactive filter state; the caller
 * invokes refresh() to (re)fetch. URLs are relative ("api/runs") so the SPA works
 * under any reverse-proxy prefix set via <base href>.
 */
import { ref, type Ref } from 'vue'

export interface ApiState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

export function useApi<T>(path: () => string): ApiState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)
  let generation = 0

  async function refresh() {
    const gen = ++generation
    loading.value = true
    error.value = null
    try {
      const resp = await fetch(path())
      if (gen !== generation) return
      if (!resp.ok) {
        throw new Error(`${resp.status} ${resp.statusText}`)
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

export async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(path)
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`)
  }
  return (await resp.json()) as T
}

export async function apiPost<T>(path: string): Promise<T> {
  const resp = await fetch(path, { method: 'POST' })
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText}`)
  }
  return (await resp.json()) as T
}
