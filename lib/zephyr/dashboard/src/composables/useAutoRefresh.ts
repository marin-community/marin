import { onUnmounted, ref } from 'vue'

export function useAutoRefresh(refresh: () => Promise<void> | void, intervalMs = 5_000) {
  const active = ref(true)
  let timer: ReturnType<typeof setInterval> | null = null

  function install() {
    if (timer !== null) clearInterval(timer)
    if (!active.value || document.visibilityState === 'hidden') return
    timer = setInterval(refresh, intervalMs)
  }

  function onVisibilityChange() {
    if (document.visibilityState === 'visible' && active.value) void refresh()
    install()
  }

  function toggle() {
    active.value = !active.value
    install()
  }

  document.addEventListener('visibilitychange', onVisibilityChange)
  install()
  onUnmounted(() => {
    if (timer !== null) clearInterval(timer)
    document.removeEventListener('visibilitychange', onVisibilityChange)
  })

  return { active, toggle }
}
