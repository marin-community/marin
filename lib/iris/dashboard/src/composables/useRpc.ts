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

const CONTROLLER_SERVICE_PATH = 'iris.cluster.ControllerService'
const WORKER_SERVICE_PATH = 'iris.cluster.WorkerService'
const LOG_SERVICE_PATH = 'proxy/system.log-server/finelog.logging.LogService'

// Cap on how much of an error body is folded into an RpcError message.
const MAX_ERROR_DETAIL_CHARS = 500

export type RpcBody = Record<string, unknown> | (() => Record<string, unknown>)

export interface RpcState<T> {
  data: Ref<T | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

/** Error thrown by RPC calls when the HTTP response is non-OK. Carries the HTTP
 *  status so callers can branch on specific failures (e.g. 404 NOT_FOUND from
 *  Connect RPC), and the server's own message as `detail`. */
export class RpcError extends Error {
  constructor(
    public readonly method: string,
    public readonly status: number,
    public readonly statusText: string,
    public readonly detail: string | null = null,
  ) {
    super(detail === null
      ? `${method}: ${status} ${statusText}`
      : `${method}: ${status} ${statusText} — ${detail}`)
    this.name = 'RpcError'
  }
}

/** Extract the server's message from a non-OK response body. The controller
 *  answers with `{"error": ...}` and Connect RPC with `{"message": ...}`;
 *  anything else surfaces as raw text. Null when the body carries nothing. */
async function readErrorDetail(resp: Response): Promise<string | null> {
  const text = (await resp.text()).trim()
  if (!text) return null
  let detail = text
  try {
    const body = JSON.parse(text) as { error?: unknown; message?: unknown }
    const message = body.error ?? body.message
    if (typeof message === 'string' && message) detail = message
  } catch {
    // Body is not JSON; surface it raw.
  }
  return detail.slice(0, MAX_ERROR_DETAIL_CHARS)
}

/** Send the browser to the iris login page on an iris auth challenge.
 *  Only iris itself answers 401 here: the endpoint proxy reports a rejected
 *  upstream as a 502, so a proxied service can never trigger this. */
function handleUnauthorized(resp: Response): void {
  if (resp.status === 401) {
    window.dispatchEvent(new CustomEvent('iris-auth-required'))
  }
}

function useRpc<T>(service: string, method: string, body?: RpcBody): RpcState<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const loading = ref(false)
  const error = ref<string | null>(null)
  let generation = 0

  // A superseded request still reports an auth challenge (rpcCall calls
  // handleUnauthorized), but never writes data or error: only the newest
  // generation owns the reactive state.
  async function refresh() {
    const gen = ++generation
    loading.value = true
    error.value = null
    try {
      const payload = await rpcCall<T>(service, method, typeof body === 'function' ? body() : body)
      if (gen !== generation) return  // superseded by a newer refresh()
      data.value = payload
    } catch (e) {
      if (gen !== generation) return  // superseded by a newer refresh()
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (gen === generation) {
        loading.value = false
      }
    }
  }

  return { data, loading, error, refresh }
}

/** RPC composable for ControllerService endpoints. */
export function useControllerRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>(CONTROLLER_SERVICE_PATH, method, body)
}

/** RPC composable for WorkerService endpoints. */
export function useWorkerRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>(WORKER_SERVICE_PATH, method, body)
}

/** POST a Connect RPC to `/<service>/<method>` and decode its response.
 *  Throws RpcError carrying the server's message on a non-OK response, and
 *  sends the browser to the login page on an iris auth challenge. */
async function rpcCall<T>(service: string, method: string, body?: Record<string, unknown>): Promise<T> {
  const resp = await fetch(`/${service}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  handleUnauthorized(resp)
  if (!resp.ok) throw new RpcError(method, resp.status, resp.statusText, await readErrorDetail(resp))
  return resp.json() as Promise<T>
}

/** One-shot RPC call for ControllerService. */
export function controllerRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  return rpcCall<T>(CONTROLLER_SERVICE_PATH, method, body)
}

/** RPC composable for LogService endpoints. */
export function useLogServiceRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>(LOG_SERVICE_PATH, method, body)
}

/** RPC composable for StatsService endpoints. */
export function useStatsRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>('iris.stats.StatsService', method, body)
}

/**
 * RPC composable for the finelog StatsService routed via the controller's
 * endpoint proxy at /proxy/system.log-server/finelog.stats.StatsService/<Method>.
 */
export function useLogServerStatsRpc<T>(
  method: string,
  body?: RpcBody,
): RpcState<T> {
  return useRpc<T>('proxy/system.log-server/finelog.stats.StatsService', method, body)
}

/** One-shot RPC call for LogService. */
export function logServiceRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  return rpcCall<T>(LOG_SERVICE_PATH, method, body)
}

/** One-shot RPC call for WorkerService. */
export function workerRpcCall<T>(method: string, body?: Record<string, unknown>): Promise<T> {
  return rpcCall<T>(WORKER_SERVICE_PATH, method, body)
}
