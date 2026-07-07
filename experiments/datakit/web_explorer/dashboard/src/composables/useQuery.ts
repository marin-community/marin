import { computed, ref } from 'vue'
import { checkDuckyStatus } from './duckyStatus'

export type Phase = 'idle' | 'submitting' | 'running' | 'done'

const POLL_MS = 700

/** Submit a named dashboard view and poll for its result, mirroring the async
 * server API (`POST api/query {view, params}` → `query_id`, then
 * `GET api/result/{id}` until terminal). URLs are relative so they resolve under
 * the controller proxy's `/proxy/datakit_explorer/` prefix. */
export async function runQuery<T>(view: string, params: Record<string, unknown>): Promise<T> {
  try {
    const resp = await fetch('api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ view, params }),
    })
    const data = await resp.json()
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`)
    for (;;) {
      const pollResp = await fetch(`api/result/${data.query_id}`)
      const poll = await pollResp.json()
      if (!pollResp.ok) throw new Error(poll.error || `HTTP ${pollResp.status}`)
      if (poll.status === 'done') return poll.result as T
      if (poll.status === 'error') throw new Error(poll.error)
      await new Promise((r) => setTimeout(r, POLL_MS))
    }
  } catch (e) {
    // A failed view often means the (preemptible) ducky backend is down;
    // refresh the banner immediately instead of waiting for the next poll.
    void checkDuckyStatus()
    throw e
  }
}

/** Reactive wrapper over `runQuery` for a single view whose result drives a section. */
export function useQuery<T>() {
  const phase = ref<Phase>('idle')
  const error = ref('')
  const result = ref<T | null>(null)
  const running = computed(() => phase.value === 'submitting' || phase.value === 'running')

  async function run(view: string, params: Record<string, unknown>): Promise<void> {
    phase.value = 'running'
    error.value = ''
    result.value = null
    try {
      result.value = await runQuery<T>(view, params)
      phase.value = 'done'
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      phase.value = 'idle'
    }
  }

  return { phase, error, result, running, run }
}
