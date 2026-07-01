import { computed, ref } from 'vue'

export type Phase = 'idle' | 'submitting' | 'running' | 'done'

export interface QueryResult {
  columns: string[]
  rows: unknown[][]
  total_rows: number
  truncated: boolean
  result_path: string
  cached: boolean
  elapsed_ms: number
  result_bytes: number
}

const POLL_MS = 1000

/** Submit SQL and poll for its result, mirroring the async server API
 * (`POST query` → `query_id`, then `GET result/{id}` until terminal). URLs are
 * relative so they resolve under the controller proxy's `/proxy/ducky/` prefix. */
export function useQuery() {
  const phase = ref<Phase>('idle')
  const error = ref('')
  const result = ref<QueryResult | null>(null)
  const running = computed(() => phase.value === 'submitting' || phase.value === 'running')

  async function poll(queryId: string): Promise<void> {
    phase.value = 'running'
    const resp = await fetch(`result/${queryId}`)
    const data = await resp.json()
    if (!resp.ok) {
      error.value = data.error || `HTTP ${resp.status}`
      phase.value = 'idle'
      return
    }
    if (data.status === 'running') {
      await new Promise((r) => setTimeout(r, POLL_MS))
      return poll(queryId)
    }
    if (data.status === 'error') {
      error.value = data.error
      phase.value = 'idle'
      return
    }
    result.value = data as QueryResult
    phase.value = 'done'
  }

  async function run(sql: string): Promise<void> {
    const trimmed = sql.trim()
    if (!trimmed || running.value) return
    phase.value = 'submitting'
    error.value = ''
    result.value = null
    try {
      const resp = await fetch('query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sql: trimmed }),
      })
      const data = await resp.json()
      if (!resp.ok) {
        error.value = data.error || `HTTP ${resp.status}`
        phase.value = 'idle'
        return
      }
      await poll(data.query_id)
    } catch (e) {
      error.value = String(e)
      phase.value = 'idle'
    }
  }

  return { phase, error, result, running, run }
}
