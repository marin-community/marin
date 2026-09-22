// Map a [0,1] score to the validated blue sequential ramp (--c-score-1..7). The rail
// draws solid fills at these steps; the matrix uses a scaled tint of the same hue so
// its cell text stays in the readable ink token in both themes.

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

export function scoreColor(value: number): string {
  const step = Math.round(clamp01(value) * 6) + 1
  return `var(--c-score-${step})`
}

export function scoreTint(value: number): string {
  const pct = Math.round(12 + clamp01(value) * 56)
  return `color-mix(in srgb, var(--c-score-5) ${pct}%, transparent)`
}
