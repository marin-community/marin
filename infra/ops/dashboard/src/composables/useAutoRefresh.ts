import { onMounted, onUnmounted } from 'vue'

const DEFAULT_REFRESH_INTERVAL = 5_000

export function useAutoRefresh(refresh: () => Promise<void>, interval = DEFAULT_REFRESH_INTERVAL) {
  let timer: number | undefined

  onMounted(() => {
    void refresh()
    timer = window.setInterval(() => {
      if (!document.hidden) void refresh()
    }, interval)
  })

  onUnmounted(() => {
    if (timer !== undefined) window.clearInterval(timer)
  })
}
