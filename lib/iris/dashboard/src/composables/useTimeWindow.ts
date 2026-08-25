/**
 * The lower time bound of a log panel: a relative preset or an absolute instant,
 * never both.
 *
 * A preset resolves against the clock on every read, so a window like "last 15m"
 * stays anchored to now across auto-refresh polls rather than to the moment the
 * preset was picked. An absolute bound is held as epoch ms, so switching the
 * panel between local time and UTC re-labels the bound without moving it.
 */
import { computed, type ComputedRef, type Ref, ref } from 'vue'

/** Relative windows offered by the time selector. 0 = no lower bound. */
export const SINCE_PRESETS: { label: string; ms: number }[] = [
  { label: 'All time', ms: 0 },
  { label: 'Last 15m', ms: 15 * 60_000 },
  { label: 'Last 1h', ms: 60 * 60_000 },
  { label: 'Last 6h', ms: 6 * 3_600_000 },
  { label: 'Last 24h', ms: 24 * 3_600_000 },
  { label: 'Last 7d', ms: 7 * 86_400_000 },
]

export interface TimeWindow {
  /** Width of the relative window in ms; 0 when there is no lower bound. */
  presetMs: Ref<number>
  /** Absolute lower bound as epoch ms; null while a preset is in effect. */
  sinceInstant: Ref<number | null>
  /** Whether the bound is an absolute instant rather than a relative preset. */
  absolute: ComputedRef<boolean>
  /** The bound as epoch ms, or undefined when unbounded. */
  sinceMs: () => number | undefined
  setSinceMs: (ms: number) => void
  selectPreset: (ms: number) => void
}

export function useTimeWindow(): TimeWindow {
  const presetMs = ref(0)
  const sinceInstant = ref<number | null>(null)

  const absolute = computed(() => sinceInstant.value !== null)

  function sinceMs(): number | undefined {
    if (sinceInstant.value !== null) return sinceInstant.value
    return presetMs.value > 0 ? Date.now() - presetMs.value : undefined
  }

  function setSinceMs(ms: number) {
    sinceInstant.value = ms
  }

  // The two forms are mutually exclusive: picking a window drops the instant.
  function selectPreset(ms: number) {
    sinceInstant.value = null
    presetMs.value = ms
  }

  return { presetMs, sinceInstant, absolute, sinceMs, setSinceMs, selectPreset }
}
