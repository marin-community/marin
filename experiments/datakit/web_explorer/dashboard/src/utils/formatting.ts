/** Locale-formatted number, em-dash for missing. */
export function fmt(n: number | null | undefined): string {
  return n == null ? '—' : Number(n).toLocaleString()
}

/** Compact number: 1.2K / 3.4M / 5B. */
export function cnum(n: number | null | undefined): string {
  if (n == null) return '—'
  const v = +n
  const a = Math.abs(v)
  if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1) + 'B'
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M'
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'K'
  return String(v)
}

/** Fraction as a percentage: 0.123 → "12.3%". */
export function pct(v: number | null | undefined): string {
  return v == null ? '—' : (100 * v).toFixed(1) + '%'
}

export const DEFAULT_SEED = 7

export function randomSeed(): number {
  return Math.floor(Math.random() * 1e6)
}
