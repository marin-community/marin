const DATE_TIME_FORMAT = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' })
const RIGGING_LOG_PREFIX = /^[DIWEC?]\d{8} \d{2}:\d{2}:\d{2} \d+ \S+ /

export function formatTimestamp(value: string): string {
  return DATE_TIME_FORMAT.format(new Date(value))
}

export function formatEpochSeconds(value: number): string {
  return DATE_TIME_FORMAT.format(new Date(value * 1_000))
}

export function formatLogMessage(value: string): string {
  return value.replace(RIGGING_LOG_PREFIX, '')
}
