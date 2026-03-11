/**
 * Typed RPC composable for calling Connect RPC endpoints.
 *
 * Wraps fetch() with reactive loading/error state. The caller gets back
 * { data, loading, error, refresh } and calls refresh() to trigger a fetch.
 * Initial data is null until the first successful fetch.
 */
import { ref, type Ref } from 'vue'

export interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

function useRpc<T>(service: string, method: string, body?: Record<string, unknown>): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function refresh() {
    loading.value = true
    error.value = null
    try {
      const resp = await fetch(`/${service}/${method}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body ?? {}),
      })
      if (!resp.ok) {
        throw new Error(`${method}: ${resp.status} ${resp.statusText}`)
      }
      data.value = await resp.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  return { data, loading, error, refresh }
}

/** RPC composable for ControllerService endpoints. */
export function useControllerRpc<T>(
  method: string,
  body?: Record<string, unknown>,
): RpcState<T> {
  return useRpc<T>('iris.cluster.ControllerService', method, body)
}

/** RPC composable for WorkerService endpoints. */
export function useWorkerRpc<T>(
  method: string,
  body?: Record<string, unknown>,
): RpcState<T> {
  return useRpc<T>('iris.cluster.WorkerService', method, body)
}
