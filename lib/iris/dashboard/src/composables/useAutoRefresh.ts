/**
 * Polling composable that calls a refresh function on an interval.
 *
 * Starts automatically by default. Cleans up the interval on component unmount.
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

  function start() {
    if (timerId !== null) return
    active.value = true
    timerId = setInterval(refreshFn, intervalMs)
  }

  function stop() {
    if (timerId === null) return
    clearInterval(timerId)
    timerId = null
    active.value = false
  }

  function toggle() {
    if (active.value) {
      stop()
    } else {
      start()
    }
  }

  if (autoStart) {
    start()
  }

  onUnmounted(stop)

  return { active, start, stop, toggle }
}
