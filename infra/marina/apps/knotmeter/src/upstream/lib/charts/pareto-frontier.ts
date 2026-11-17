// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export interface ParetoCoordinate {
  x: number;
  y: number;
}

/** Global, non-dominated observations, sorted by x. Input is never mutated. */
export function globalParetoFrontier<T extends ParetoCoordinate>(
  points: readonly T[],
  maximizeX: boolean,
  maximizeY: boolean,
): T[] {
  const xSign = maximizeX ? -1 : 1;
  const ySign = maximizeY ? -1 : 1;
  const sorted = points
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    .toSorted((a, b) => xSign * (a.x - b.x) || ySign * (a.y - b.y));
  let bestY = Infinity;
  return sorted
    .filter((point) => {
      const score = ySign * point.y;
      if (score >= bestY) return false;
      bestY = score;
      return true;
    })
    .toSorted((a, b) => a.x - b.x);
}
