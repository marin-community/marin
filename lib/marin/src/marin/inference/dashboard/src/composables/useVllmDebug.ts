import { ref, watch } from 'vue'
import type { VllmRequestDebug } from '../lib/vllm_debug'

export function useVllmDebug() {
  const enabled = ref(false)
  const lastRequest = ref<VllmRequestDebug | null>(null)

  watch(enabled, () => {
    lastRequest.value = null
  }, { flush: 'sync' })

  return { enabled, lastRequest }
}
