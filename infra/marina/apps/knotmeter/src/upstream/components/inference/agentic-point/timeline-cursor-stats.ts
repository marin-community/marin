// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export interface SortedRequestTimes {
  credits: number[];
  starts: number[];
  ends: number[];
}

export interface CursorStats {
  running: number;
  waiting: number;
  completed: number;
  inflight: number;
}

export function countLeq(sorted: number[], target: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid]! <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function countLt(sorted: number[], target: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid]! < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function cursorStatsAt(
  times: SortedRequestTimes,
  t: number,
): CursorStats {
  const startsLeq = countLeq(times.starts, t);
  const running = Math.max(0, startsLeq - countLt(times.ends, t));
  const waiting = Math.max(0, countLeq(times.credits, t) - startsLeq);
  const completed = countLeq(times.ends, t);
  return { running, waiting, completed, inflight: running + waiting };
}
