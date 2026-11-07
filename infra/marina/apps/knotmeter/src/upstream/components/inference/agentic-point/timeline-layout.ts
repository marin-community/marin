// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export const TIMELINE_BODY_MAX_HEIGHT = 480;

export const LABEL_WIDTH = 360;
export const ROW_HEIGHT = 22;
export const ROW_GAP = 3;
export const HEADER_HEIGHT = 24;
export const PADDING_RIGHT = 12;
export const CHART_WIDTH = 920;
export const ROW_SPAN = ROW_HEIGHT + ROW_GAP;

export function timelineSvgHeight(rowCount: number): number {
  return HEADER_HEIGHT + rowCount * ROW_SPAN + 6;
}

export function visibleTimelineRowRange(
  rowCount: number,
  scrollTop: number,
  viewportHeight: number,
  overscan = 4,
): { start: number; end: number } {
  const firstVisible = Math.floor(
    Math.max(0, scrollTop - HEADER_HEIGHT) / ROW_SPAN,
  );
  const lastVisible = Math.ceil(
    Math.max(0, scrollTop + viewportHeight - HEADER_HEIGHT) / ROW_SPAN,
  );
  return {
    start: Math.min(rowCount, Math.max(0, firstVisible - overscan)),
    end: Math.min(rowCount, lastVisible + overscan),
  };
}
