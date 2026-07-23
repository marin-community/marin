/**
 * Client-side pager over a run's sample rows for one (task, filter) selection.
 *
 * Fetches 25-row pages on demand and caches them by `task|filter|page`, so scrubbing back and
 * forth over already-seen rows is instant. `ensure(i)` fetches the page containing row `i` and
 * prefetches a neighboring page once `i` is within 5 rows of a page boundary, so arrow-key
 * navigation rarely blocks on a request. The cache is dropped whenever `task` or `filter` changes.
 */
import { ref, watch, type Ref } from 'vue'
import { apiGet } from '@/composables/useApi'
import type { SampleRow, SamplesResponse } from '@/types/api'

export type SampleFilter = 'all' | 'correct' | 'incorrect'

const PAGE_SIZE = 25
const PREFETCH_MARGIN = 5

export interface SamplePager {
  total: Ref<number>
  counts: Ref<{ all: number; correct: number; incorrect: number } | null>
  primaryMetric: Ref<string | null>
  loading: Ref<boolean>
  error: Ref<string | null>
  sample: (i: number) => SampleRow | null
  ensure: (i: number) => void
}

export function useSamplePager(runId: string, task: Ref<string>, filter: Ref<SampleFilter>): SamplePager {
  const total = ref(0)
  const counts = ref<{ all: number; correct: number; incorrect: number } | null>(null)
  const primaryMetric = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  // Bumped on every page landing in the (plain, non-reactive) `pages` cache so `sample()` can be
  // called from a Vue computed and re-evaluate once the page it needs has arrived.
  const revision = ref(0)

  let pages = new Map<string, SampleRow[]>()
  let pending = new Set<string>()
  let generation = 0

  function pageKey(page: number): string {
    return `${task.value}|${filter.value}|${page}`
  }

  function invalidate() {
    generation++
    pages = new Map()
    pending = new Set()
    total.value = 0
    counts.value = null
    revision.value++
  }

  watch([task, filter], invalidate)

  async function fetchPage(page: number) {
    if (!task.value || page < 0) return
    const key = pageKey(page)
    if (pages.has(key) || pending.has(key)) return
    const gen = generation
    const pendingSet = pending
    const pagesMap = pages
    pendingSet.add(key)
    loading.value = true
    try {
      const params = new URLSearchParams({
        task: task.value,
        offset: String(page * PAGE_SIZE),
        limit: String(PAGE_SIZE),
        correct: filter.value,
      })
      const resp = await apiGet<SamplesResponse>(`api/runs/${runId}/samples?${params.toString()}`)
      if (gen !== generation) return
      pagesMap.set(key, resp.rows)
      revision.value++
      total.value = resp.total
      primaryMetric.value = resp.primary_metric
      if (resp.counts) counts.value = resp.counts
      error.value = null
    } catch (e) {
      if (gen !== generation) return
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      pendingSet.delete(key)
      if (gen === generation) loading.value = pendingSet.size > 0
    }
  }

  function ensure(i: number) {
    if (i < 0) return
    const page = Math.floor(i / PAGE_SIZE)
    fetchPage(page)
    const withinPage = i - page * PAGE_SIZE
    if (withinPage < PREFETCH_MARGIN) fetchPage(page - 1)
    if (PAGE_SIZE - withinPage <= PREFETCH_MARGIN) fetchPage(page + 1)
  }

  function sample(i: number): SampleRow | null {
    void revision.value
    if (i < 0) return null
    const page = Math.floor(i / PAGE_SIZE)
    const rows = pages.get(pageKey(page))
    if (!rows) return null
    return rows[i - page * PAGE_SIZE] ?? null
  }

  return { total, counts, primaryMetric, loading, error, sample, ensure }
}
