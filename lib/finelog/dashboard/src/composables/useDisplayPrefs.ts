/**
 * Viewer display preferences, shared by every component and persisted across
 * reloads.
 *
 * The state is module-level rather than provide/inject so a timestamp in the
 * results table, the log stream, and a chart axis always agree — disagreeing
 * clocks on one screen is worse than either choice on its own.
 */
import { ref, watch } from 'vue'

export type TimeZoneMode = 'local' | 'utc' | 'raw'

const STORAGE_KEY = 'finelog-timezone-mode'

function initialMode(): TimeZoneMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'local' || stored === 'utc' || stored === 'raw') return stored
  } catch {
    // localStorage throws rather than returning null when the browser blocks
    // site data. A display preference is not worth failing the app over, so
    // fall through to the default and leave the session unpersisted.
  }
  // UTC by default: finelog rows come from a fleet spanning several regions and
  // its own logs are UTC, so UTC is the zone that matches what an operator is
  // comparing against.
  return 'utc'
}

export const timeZoneMode = ref<TimeZoneMode>(initialMode())

watch(timeZoneMode, (mode) => {
  try {
    localStorage.setItem(STORAGE_KEY, mode)
  } catch {
    // Blocked site data or a full quota: the preference still applies for this
    // session, it just will not survive a reload.
  }
})
