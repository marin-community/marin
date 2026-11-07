// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export function lttbDownsample<T>(
  data: T[],
  target: number,
  getX: (d: T) => number,
  getY: (d: T) => number,
): T[] {
  if (data.length <= target) return data;
  if (target < 3) return [data[0], data.at(-1)!];

  const result: T[] = [data[0]];
  const bucketSize = (data.length - 2) / (target - 2);

  let prevIndex = 0;

  for (let i = 0; i < target - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.min(
      Math.floor((i + 2) * bucketSize) + 1,
      data.length - 1,
    );

    const nextBucketStart = bucketEnd;
    const nextBucketEnd = Math.min(
      Math.floor((i + 3) * bucketSize) + 1,
      data.length - 1,
    );
    let avgX = 0;
    let avgY = 0;
    const nextBucketLen = nextBucketEnd - nextBucketStart + 1;
    for (let j = nextBucketStart; j <= nextBucketEnd; j++) {
      avgX += getX(data[j]);
      avgY += getY(data[j]);
    }
    avgX /= nextBucketLen;
    avgY /= nextBucketLen;

    const prevX = getX(data[prevIndex]);
    const prevY = getY(data[prevIndex]);
    let maxArea = -1;
    let bestIndex = bucketStart;

    for (let j = bucketStart; j < bucketEnd; j++) {
      const area = Math.abs(
        (prevX - avgX) * (getY(data[j]) - prevY) -
          (prevX - getX(data[j])) * (avgY - prevY),
      );
      if (area > maxArea) {
        maxArea = area;
        bestIndex = j;
      }
    }

    result.push(data[bestIndex]);
    prevIndex = bestIndex;
  }

  result.push(data.at(-1)!);
  return result;
}
