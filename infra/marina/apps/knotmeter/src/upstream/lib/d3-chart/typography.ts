// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export const CHART_TYPE = {
  axisLabel: 12,
  axisLabelSub: 10,
  annotation: 11,
  annotationSub: 10,
  dataLabel: 10,
  lineLabel: 13,
  micro: 9,
  watermark: 20,
} as const;

export function px(size: number): string {
  return `${size}px`;
}

export const CHART_FONT_SANS =
  'var(--font-dm-sans), -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
export const CHART_FONT_MINECRAFT =
  'var(--font-minecraft), "Monocraft", monospace';
