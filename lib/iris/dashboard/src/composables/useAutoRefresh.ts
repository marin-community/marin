/**
 * Polling composable that calls a refresh function on an interval.
 *
 * Starts automatically by default. Cleans up the interval on component unmount.
 *
 * Polling is automatically paused while the document is hidden
 * (`document.visibilityState === 'hidden'`) so that background tabs don't
 * hammer the controller. When the tab becomes visible again an immediate
 * refresh is issued and the interval resumes. The `active` ref reflects the
 * user-requested state, not the visibility-driven pause: a component that
 * wants to reflect "currently polling" should combine `active` with the
 * document visibility itself.
 */
import { ref, onUnmounted } from 'vue'

export interface AutoRefreshState {
  active: Readonly<ReturnType<typeof ref<boolean>>>
  start: () => void
  stop: () => void
  toggle: () => void
}

export function useAutoRefresh(
  refreshFn: () => Promise<void> | void,
  intervalMs: number,
  autoStart = true,
): AutoRefreshState {
  const active = ref(false)
  let timerId: ReturnType<typeof setInterval> | null = null

  function clearTimer() {
    if (timerId !== null) {
      clearInterval(timerId)
      timerId = null
    }
  }

  function installTimer() {
    clearTimer()
    if (!active.value) return
    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return
    timerId = setInterval(refreshFn, intervalMs)
  }

  function start() {
    if (active.value) return
    active.value = true
    installTimer()
  }

  function stop() {
    active.value = false
    clearTimer()
  }

  function toggle() {
    if (active.value) {
      stop()
    } else {
      start()
    }
  }

  function onVisibilityChange() {
    if (!active.value) return
    if (document.visibilityState === 'hidden') {
      clearTimer()
      return
    }
    // Tab became visible: catch up once, then resume the interval.
    void refreshFn()
    installTimer()
  }

  if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', onVisibilityChange)
  }

  if (autoStart) {
    start()
  }

  onUnmounted(() => {
    clearTimer()
    active.value = false
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  })

  return { active, start, stop, toggle }
}
