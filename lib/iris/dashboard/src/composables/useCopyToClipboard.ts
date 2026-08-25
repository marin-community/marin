import { ref } from 'vue'

// How long the `copied`/`error` flags stay set before reverting, so a button can
// flash "Copied" and settle back to its idle label.
const RESET_DELAY_MS = 1500

/**
 * Write text to the clipboard and expose transient `copied`/`error` flags that
 * auto-reset, for the common "flash a confirmation on a button" pattern.
 */
export function useCopyToClipboard() {
  const copied = ref(false)
  const error = ref(false)
  let timer: ReturnType<typeof setTimeout> | undefined

  async function copy(text: string): Promise<void> {
    error.value = false
    try {
      await navigator.clipboard.writeText(text)
      copied.value = true
    } catch {
      error.value = true
    }
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => { copied.value = false; error.value = false }, RESET_DELAY_MS)
  }

  return { copied, error, copy }
}
