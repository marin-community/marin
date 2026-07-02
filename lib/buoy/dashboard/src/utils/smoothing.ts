// wandb-style debiased EMA smoothing.
export function ema(values: number[], w: number): number[] {
  if (w <= 0) return values
  let last = 0
  let debias = 0
  const out: number[] = []
  for (const v of values) {
    last = last * w + (1 - w) * v
    debias = debias * w + (1 - w)
    out.push(last / debias)
  }
  return out
}
