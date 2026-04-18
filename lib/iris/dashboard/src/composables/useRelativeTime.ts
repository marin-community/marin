/**
 * Reactive relative time formatting composable.
 *
 * Takes a timestamp (epoch ms as number or string) and returns a ref
 * that updates every ~30 seconds with a human-readable relative string
 * like "5s ago", "3m ago", "2h ago", "1d ago".
 */
import { ref, watch, onUnmounted, type Ref, toValue, type MaybeRefOrGetter } from 'vue'
import { formatRelativeTime } from '@/utils/formatting'

const UPDATE_INTERVAL_MS = 30_000

function parseTimestamp(ts: string | number | null | undefined): number {
  if (ts === null || ts === undefined) return 0
  if (typeof ts === 'number') return ts
  return parseInt(ts, 10) || 0
}

export function useRelativeTime(
  timestamp: MaybeRefOrGetter<string | number | null | undefined>,
): Ref<string> {
  const display = ref(formatRelativeTime(parseTimestamp(toValue(timestamp))))

  function update() {
    display.value = formatRelativeTime(parseTimestamp(toValue(timestamp)))
  }

  watch(() => toValue(timestamp), update)

  const timerId = setInterval(update, UPDATE_INTERVAL_MS)
  onUnmounted(() => clearInterval(timerId))

  return display
}
