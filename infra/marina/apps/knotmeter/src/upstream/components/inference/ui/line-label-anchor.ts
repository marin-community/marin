// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export const pointNearestX = <T extends { x: number }>(
  pts: readonly T[],
  targetX: number,
): T => {
  let best = pts[0];
  let bestDist = Math.abs(pts[0].x - targetX);
  for (let i = 1; i < pts.length; i++) {
    const dist = Math.abs(pts[i].x - targetX);
    if (dist < bestDist) {
      bestDist = dist;
      best = pts[i];
    }
  }
  return best;
};
