/**
 * Typed RPC composable for calling Connect RPC endpoints.
 *
 * Wraps fetch() with reactive loading/error state. The caller gets back
 * { data, loading, error, refresh } and calls refresh() to trigger a fetch.
 * Initial data is null until the first successful fetch.
 *
 * The body parameter can be a static object or a factory function for cases
 * where request parameters depend on reactive state (e.g. props, pagination).
 */
import { ref, type Ref } from 'vue'

export type RpcBody = Record<string, unknown> | (() => Record<string, unknown>)

export interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

const AUTH_TOKEN_KEY = 'iris-auth-token'

export function getAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setAuthToken(token: string): void {
  localStorage.setItem(AUTH_TOKEN_KEY, token)
}

export function clearAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY)
}

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = getAuthToken()
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

function handleUnauthorized(resp: Response): void {
  if (resp.status === 401) {
    clearAuthToken()
    window.dispatchEvent(new CustomEvent('iris-auth-required'))
  }
}

function useRpc<T>(service: string, method: string, body?: RpcBody): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function refresh() {
    loading.value = true
    error.value = null
    try {
      const resolvedBody = typeof body === 'function' ? body() : (body ?? {})
      const resp = await fetch(`/${service}/${method}`, {
        method: 'POST',
        headers: authHeaders(),
        body: JSON.stringify(resolvedBody),
      })
      handleUnauthorized(resp)
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
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>('iris.cluster.ControllerService', method, body)
}

/** RPC composable for WorkerService endpoints. */
export function useWorkerRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>('iris.cluster.WorkerService', method, body)
}

/** One-shot RPC call returning a Promise. For use in async functions that
 *  need to call multiple RPCs or handle the response imperatively. */
export async function controllerRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  const resp = await fetch(`/iris.cluster.ControllerService/${method}`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body ?? {}),
  })
  handleUnauthorized(resp)
  if (!resp.ok) throw new Error(`${method}: ${resp.status} ${resp.statusText}`)
  return resp.json() as Promise<T>
}

export async function workerRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  const resp = await fetch(`/iris.cluster.WorkerService/${method}`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body ?? {}),
  })
  handleUnauthorized(resp)
  if (!resp.ok) throw new Error(`${method}: ${resp.status} ${resp.statusText}`)
  return resp.json() as Promise<T>
}
