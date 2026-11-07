// SPDX-License-Identifier: GPL-3.0-only
// InferenceX 315128ce Pareto and monotone Hermite calculations; see THIRD_PARTY.md.
export function paretoFrontUpperLeft<T>(
  points: T[],
  getX: (p: T) => number,
  getY: (p: T) => number,
): T[] {
  if (points.length === 0) return [];

  const sorted = [...points].toSorted((a, b) => {
    const ax = getX(a);
    const bx = getX(b);
    if (ax === bx) return getY(b) - getY(a);
    return ax - bx;
  });

  const front: T[] = [];

  for (const point of sorted) {
    const px = getX(point);
    const py = getY(point);

    if (front.length > 0 && getX(front.at(-1)!) === px) {
      if (py > getY(front.at(-1)!)) {
        front[front.length - 1] = point;
      }
      continue;
    }

    while (front.length > 0 && py >= getY(front.at(-1)!)) {
      front.pop();
    }
    front.push(point);
  }

  return front;
}

export function sign(x: number): number {
  return x < 0 ? -1 : 1;
}

export function monotoneSlopes(xs: number[], ys: number[]): number[] {
  const n = xs.length;
  if (n < 2) return Array.from({ length: n }, () => 0);

  const h: number[] = [];
  const s: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    const hi = xs[i + 1] - xs[i];
    h.push(hi);
    s.push(hi === 0 ? 0 : (ys[i + 1] - ys[i]) / hi);
  }

  const m: number[] = Array.from({ length: n }, () => 0);
  for (let i = 1; i < n - 1; i++) {
    const s0 = s[i - 1];
    const s1 = s[i];
    const h0 = h[i - 1];
    const h1 = h[i];
    const p = (s0 * h1 + s1 * h0) / (h0 + h1);
    m[i] =
      (sign(s0) + sign(s1)) *
        Math.min(Math.abs(s0), Math.abs(s1), 0.5 * Math.abs(p)) || 0;
  }

  m[0] = h[0] ? (3 * s[0] - m[1]) / 2 : m[1];
  m[n - 1] = h[n - 2] ? (3 * s[n - 2] - m[n - 2]) / 2 : m[n - 2];

  return m;
}

export function hermiteInterpolate(
  xs: number[],
  ys: number[],
  m: number[],
  targetX: number,
): number {
  const n = xs.length;
  if (n === 0) return 0;
  if (n === 1) return ys[0];

  if (targetX <= xs[0]) return ys[0];
  if (targetX >= xs[n - 1]) return ys[n - 1];

  let lo = 0;
  let hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= targetX) lo = mid;
    else hi = mid;
  }

  const hh = xs[hi] - xs[lo];
  if (hh === 0) return ys[lo];

  const t = (targetX - xs[lo]) / hh;
  const t2 = t * t;
  const t3 = t2 * t;

  const h00 = 2 * t3 - 3 * t2 + 1;
  const h10 = t3 - 2 * t2 + t;
  const h01 = -2 * t3 + 3 * t2;
  const h11 = t3 - t2;

  return h00 * ys[lo] + h10 * hh * m[lo] + h01 * ys[hi] + h11 * hh * m[hi];
}
