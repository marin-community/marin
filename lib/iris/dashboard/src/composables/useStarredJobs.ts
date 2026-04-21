/**
 * Starred-jobs composable.
 *
 * Groups all state and side-effects for the Jobs tab's "starred" feature:
 *   - the persisted set of starred job IDs (localStorage-backed)
 *   - the per-ID fetched job data (via GetJobStatus), since ListJobs has no
 *     `jobId IN (...)` filter and starred jobs outside the current page
 *     would otherwise disappear
 *   - loading / error / rate-limit-notice state
 *   - toggle + fetch helpers
 *
 * The "show starred only" toggle is kept here too so the whole concept lives
 * behind one object; the caller is responsible for URL-syncing it.
 *
 * Returns a `reactive` object so callers access fields as plain values
 * (`starred.ids`, `starred.showOnly`, …) in both script and template. Watch
 * sources should use a getter, e.g. `watch(() => starred.showOnly, …)`.
 */
import { reactive } from 'vue'
import { controllerRpcCall } from '@/composables/useRpc'
import type { JobStatus, GetJobStatusResponse } from '@/types/rpc'

export interface StarredJobsOptions {
  /** localStorage key for persisting the starred ID set. */
  storageKey: string
  /** Maximum number of jobs that may be starred at once. */
  maxCount: number
  /** Initial value of `showOnly` (e.g. hydrated from the URL). */
  initialShowOnly?: boolean
  /** How long the "limit reached" notice stays visible. */
  limitNoticeDurationMs?: number
}

export interface StarredJobs {
  ids: Set<string>
  showOnly: boolean
  jobs: JobStatus[]
  loading: boolean
  error: string | null
  limitNotice: string | null
  readonly maxCount: number
  /** Star or unstar a job. Silently no-ops with a notice if at the limit. */
  toggle: (job: JobStatus) => void
  /** Fetch fresh data for every currently-starred ID. */
  fetch: () => Promise<void>
}

export function useStarredJobs(options: StarredJobsOptions): StarredJobs {
  const { storageKey, maxCount, initialShowOnly = false, limitNoticeDurationMs = 4000 } = options

  function loadIds(): Set<string> {
    try {
      const stored = localStorage.getItem(storageKey)
      return stored ? new Set(JSON.parse(stored) as string[]) : new Set()
    } catch {
      return new Set()
    }
  }

  function saveIds() {
    try {
      localStorage.setItem(storageKey, JSON.stringify([...state.ids]))
    } catch {
      // ignore
    }
  }

  const state: StarredJobs = reactive({
    ids: loadIds(),
    showOnly: initialShowOnly,
    jobs: [] as JobStatus[],
    loading: false,
    error: null as string | null,
    limitNotice: null as string | null,
    maxCount,

    toggle(job: JobStatus) {
      const next = new Set(state.ids)
      if (next.has(job.jobId)) {
        next.delete(job.jobId)
      } else {
        if (next.size >= maxCount) {
          state.limitNotice = `You can star at most ${maxCount} jobs — unstar one first.`
          setTimeout(() => { state.limitNotice = null }, limitNoticeDurationMs)
          return
        }
        next.add(job.jobId)
      }
      state.ids = next
      saveIds()
    },

    // Fetch each starred job individually — the ListJobs RPC does not support
    // filtering by a set of job IDs, so this is the simplest correct way to
    // show only starred jobs without losing any due to pagination.
    async fetch() {
      const currentIds = [...state.ids]
      if (currentIds.length === 0) {
        state.jobs = []
        state.error = null
        return
      }
      state.loading = true
      state.error = null
      try {
        const results = await Promise.allSettled(
          currentIds.map(id => controllerRpcCall<GetJobStatusResponse>('GetJobStatus', { jobId: id })),
        )
        state.jobs = results
          .filter((r): r is PromiseFulfilledResult<GetJobStatusResponse> => r.status === 'fulfilled' && !!r.value?.job)
          .map(r => r.value.job)
        const failures = results.filter(r => r.status === 'rejected').length
        if (failures > 0 && state.jobs.length === 0) {
          state.error = `Failed to load ${failures} starred job${failures !== 1 ? 's' : ''}`
        }
      } finally {
        state.loading = false
      }
    },
  })

  return state
}
