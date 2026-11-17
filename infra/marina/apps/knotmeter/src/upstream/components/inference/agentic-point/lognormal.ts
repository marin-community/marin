// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export interface LogHistogram {
  edges: number[];
  counts: number[];
  lnStep: number;
  lnMin: number;
  lnMax: number;
}

export function positiveValues(values: readonly number[]): number[] {
  return values.filter((v) => Number.isFinite(v) && v > 0);
}

export function logHistogram(
  values: readonly number[],
  bins: number,
): LogHistogram | null {
  const positive = positiveValues(values);
  if (positive.length === 0) return null;

  const nBins = Math.max(1, Math.floor(bins));
  let min = positive[0]!;
  let max = min;
  for (let i = 1; i < positive.length; i++) {
    const value = positive[i]!;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  let lnMin = Math.log(min);
  let lnMax = Math.log(max);
  if (!(lnMax > lnMin)) {
    lnMin -= Math.LN2;
    lnMax += Math.LN2;
  }

  const lnStep = (lnMax - lnMin) / nBins;
  const counts: number[] = Array.from({ length: nBins }, () => 0);
  for (const value of positive) {
    const index = Math.min(
      nBins - 1,
      Math.floor((Math.log(value) - lnMin) / lnStep),
    );
    counts[index]!++;
  }

  const edges = Array.from({ length: nBins + 1 }, (_, i) =>
    Math.exp(lnMin + i * lnStep),
  );
  return { edges, counts, lnStep, lnMin, lnMax };
}

export function logTicks(min: number, max: number): number[] {
  if (!(min > 0) || !(max > min)) return [];
  const decades = Math.log10(max) - Math.log10(min);
  const multipliers = decades > 3 ? [1] : decades > 1.5 ? [1, 5] : [1, 2, 5];
  const candidates: number[] = [min];
  for (
    let exp = Math.floor(Math.log10(min));
    exp <= Math.ceil(Math.log10(max));
    exp++
  ) {
    for (const multiplier of multipliers) {
      const tick = multiplier * 10 ** exp;
      if (tick > min && tick < max) candidates.push(tick);
    }
  }
  candidates.push(max);

  const minGap = decades * 0.06;
  const kept: number[] = [];
  for (const tick of candidates) {
    const previous = kept.at(-1);
    if (
      previous === undefined ||
      Math.log10(tick) - Math.log10(previous) >= minGap
    ) {
      kept.push(tick);
    }
  }
  if (kept.at(-1) !== max) {
    if (kept.length > 1) kept.pop();
    kept.push(max);
  }
  return kept;
}
