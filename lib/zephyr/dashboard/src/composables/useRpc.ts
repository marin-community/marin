import { ref, type Ref } from 'vue'

const SERVICE = 'zephyr.dashboard.v1.CoordinatorDashboardService'

export type RpcBody = Record<string, unknown> | (() => Record<string, unknown>)

export interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

async function errorMessage(method: string, response: Response): Promise<string> {
  const text = await response.text().catch(() => '')
  if (text) {
    try {
      const parsed = JSON.parse(text) as { message?: unknown }
      if (typeof parsed.message === 'string') return `${method}: ${parsed.message}`
    } catch {
      return `${method}: ${response.status} ${text}`
    }
  }
  return `${method}: ${response.status} ${response.statusText}`
}

export function useDashboardRpc<T>(method: string, body?: RpcBody): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)
  let generation = 0

  async function refresh() {
    const currentGeneration = ++generation
    loading.value = true
    error.value = null
    try {
      const resolvedBody = typeof body === 'function' ? body() : (body ?? {})
      const response = await fetch(`${SERVICE}/${method}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Connect-Protocol-Version': '1',
        },
        body: JSON.stringify(resolvedBody),
      })
      if (currentGeneration !== generation) return
      if (!response.ok) throw new Error(await errorMessage(method, response))
      data.value = (await response.json()) as T
    } catch (caught) {
      if (currentGeneration !== generation) return
      error.value = caught instanceof Error ? caught.message : String(caught)
    } finally {
      if (currentGeneration === generation) loading.value = false
    }
  }

  return { data, loading, error, refresh }
}
