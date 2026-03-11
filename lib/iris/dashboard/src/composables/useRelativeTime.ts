/**
 * Reactive relative time formatting composable.
 *
 * Takes a timestamp (epoch ms as number or string) and returns a ref
 * that updates every ~30 seconds with a human-readable relative string
 * like "5s ago", "3m ago", "2h ago", "1d ago".
 */
import { ref, watch, onUnmounted, type Ref, toValue, type MaybeRefOrGetter } from 'vue'

const UPDATE_INTERVAL_MS = 30_000

function formatRelative(timestampMs: number): string {
  if (!timestampMs) return '-'
  const seconds = Math.floor((Date.now() - timestampMs) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

function parseTimestamp(ts: string | number | null | undefined): number {
  if (ts === null || ts === undefined) return 0
  if (typeof ts === 'number') return ts
  return parseInt(ts, 10) || 0
}

export function useRelativeTime(
  timestamp: MaybeRefOrGetter<string | number | null | undefined>,
): Ref<string> {
  const display = ref(formatRelative(parseTimestamp(toValue(timestamp))))

  function update() {
    display.value = formatRelative(parseTimestamp(toValue(timestamp)))
  }

  watch(() => toValue(timestamp), update)

  const timerId = setInterval(update, UPDATE_INTERVAL_MS)
  onUnmounted(() => clearInterval(timerId))

  return display
}
