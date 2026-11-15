import { onMounted, onUnmounted, ref } from 'vue'
import { fetchHealth, fetchInfo } from '../lib/api'
import type { ServerStatus, ServingInfo } from '../lib/types'

const POLL_INTERVAL_MS = 15_000

/** Poll /info (web search availability can change) and /health (the status dot). */
export function useServing() {
  const info = ref<ServingInfo | null>(null)
  const status = ref<ServerStatus>('connecting')
  const model = ref('')

  async function refreshInfo() {
    try {
      info.value = await fetchInfo()
      if (info.value.model) model.value = info.value.model
    } catch {
      // /health reports reachability; retried on the poll interval until loaded.
    }
  }

  async function refreshHealth() {
    try {
      const health = await fetchHealth()
      status.value = health.ok ? 'ok' : 'loading'
      if (health.model && !model.value) model.value = health.model
    } catch {
      status.value = 'bad'
    }
  }

  onMounted(() => {
    refreshInfo()
    refreshHealth()
    const timer = setInterval(() => {
      refreshHealth()
      refreshInfo()
    }, POLL_INTERVAL_MS)
    onUnmounted(() => clearInterval(timer))
  })

  return { info, status, model }
}
