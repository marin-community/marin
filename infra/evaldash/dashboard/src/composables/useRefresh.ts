/**
 * Process-wide refresh coordination.
 *
 * The mounted view registers its data refetch via `onViewRefresh`. The header's manual
 * button and a visibility-aware timer both call `triggerRefresh`, which refetches whichever
 * view is currently mounted. The timer polls every 60s and pauses while the tab is hidden,
 * refetching once on the transition back to visible.
 *
 * `useServerRefresh` exposes one shared in-flight state for the "ingest now" action behind
 * both the header and status-page refresh buttons: a POST /api/refresh with spinner/disabled
 * state, an "updated HH:MM:SS" confirmation, and a visible error. State is module-level so the
 * two buttons stay in lockstep — starting one disables the other.
 */
import { onMounted, onUnmounted, readonly, ref, type DeepReadonly, type Ref } from 'vue'
import { apiPost } from '@/composables/useApi'

const REFRESH_INTERVAL_MS = 60_000

const listeners = new Set<() => void>()

export function triggerRefresh(): void {
  for (const listener of listeners) listener()
}

const refreshing = ref(false)
const lastRefreshAt = ref<Date | null>(null)
const refreshError = ref<string | null>(null)

export interface ServerRefresh {
  refreshing: DeepReadonly<Ref<boolean>>
  lastRefreshAt: DeepReadonly<Ref<Date | null>>
  refreshError: DeepReadonly<Ref<string | null>>
  refreshNow: () => Promise<void>
}

/** Run one server-side ingest pass, then refetch the mounted view. Shared across all callers. */
async function refreshNow(): Promise<void> {
  if (refreshing.value) return
  refreshing.value = true
  refreshError.value = null
  try {
    await apiPost('api/refresh')
    lastRefreshAt.value = new Date()
    triggerRefresh()
  } catch (e) {
    refreshError.value = e instanceof Error ? e.message : String(e)
  } finally {
    refreshing.value = false
  }
}

export function useServerRefresh(): ServerRefresh {
  return {
    refreshing: readonly(refreshing),
    lastRefreshAt: readonly(lastRefreshAt),
    refreshError: readonly(refreshError),
    refreshNow,
  }
}

/** Register `refetch` as the current view's data refresh for the component's lifetime. */
export function onViewRefresh(refetch: () => void): void {
  onMounted(() => listeners.add(refetch))
  onUnmounted(() => listeners.delete(refetch))
}

/** Drive the visibility-aware auto-refresh timer; call once from the app root. */
export function useAutoRefresh(): void {
  let timer: number | undefined

  function refreshWhenVisible() {
    if (!document.hidden) triggerRefresh()
  }

  onMounted(() => {
    timer = window.setInterval(refreshWhenVisible, REFRESH_INTERVAL_MS)
    document.addEventListener('visibilitychange', refreshWhenVisible)
  })
  onUnmounted(() => {
    if (timer !== undefined) window.clearInterval(timer)
    document.removeEventListener('visibilitychange', refreshWhenVisible)
  })
}
