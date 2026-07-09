/**
 * The lower time bound of a log panel: a relative preset or an absolute instant,
 * never both.
 *
 * A preset resolves against the clock on every read, so a window like "last 15m"
 * stays anchored to now across auto-refresh polls rather than to the moment the
 * preset was picked.
 */
import { computed, type ComputedRef, type Ref, ref, watch } from 'vue'

export type TimeZoneName = 'local' | 'utc'

/** Relative windows offered by the "Since" selector. 0 = no lower bound. */
export const SINCE_PRESETS: { label: string; ms: number }[] = [
  { label: 'All time', ms: 0 },
  { label: 'Last 15m', ms: 15 * 60_000 },
  { label: 'Last 1h', ms: 60 * 60_000 },
  { label: 'Last 6h', ms: 6 * 3_600_000 },
  { label: 'Last 24h', ms: 24 * 3_600_000 },
  { label: 'Last 7d', ms: 7 * 86_400_000 },
]

/** The synthetic selector option shown while an absolute instant is in effect. */
export const CUSTOM_PRESET = -1

export interface TimeWindow {
  /** Width of the relative window in ms; 0 when there is no lower bound. */
  presetMs: Ref<number>
  /** A `datetime-local` string; empty when a preset is in effect. */
  customSince: Ref<string>
  /** Whether the bound is an absolute instant rather than a relative preset. */
  absolute: ComputedRef<boolean>
  /** The bound as epoch ms, or undefined when unbounded. */
  sinceMs: () => number | undefined
  selectPreset: (ms: number) => void
}

/**
 * Read a `datetime-local` value (e.g. "2026-06-30T02:08") as epoch ms in `zone`.
 *
 * The value carries no timezone, so it is read in the panel's selected zone
 * rather than the browser's. Returns NaN when the value does not parse.
 */
function parseDateTimeLocal(value: string, zone: TimeZoneName): number {
  const m = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/)
  if (!m) return NaN
  const [year, month, day, hour, minute, second] = m.slice(1).map((p) => Number(p ?? 0))
  return zone === 'utc'
    ? Date.UTC(year, month - 1, day, hour, minute, second)
    : new Date(year, month - 1, day, hour, minute, second).getTime()
}

export function useTimeWindow(timeZone: Ref<TimeZoneName>): TimeWindow {
  const presetMs = ref(0)
  const customSince = ref('')

  // The two forms are mutually exclusive: naming an instant drops the preset.
  watch(customSince, (value) => {
    if (value) presetMs.value = 0
  })

  const absolute = computed(() => customSince.value !== '')

  function sinceMs(): number | undefined {
    if (customSince.value) {
      const ms = parseDateTimeLocal(customSince.value, timeZone.value)
      return Number.isNaN(ms) ? undefined : ms
    }
    return presetMs.value > 0 ? Date.now() - presetMs.value : undefined
  }

  function selectPreset(ms: number) {
    // Re-selecting the synthetic "Custom" option leaves the instant in effect.
    if (ms === CUSTOM_PRESET) return
    customSince.value = ''
    presetMs.value = ms
  }

  return { presetMs, customSince, absolute, sinceMs, selectPreset }
}
