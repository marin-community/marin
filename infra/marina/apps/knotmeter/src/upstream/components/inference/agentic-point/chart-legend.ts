// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export const LEGEND_FONT_SIZE = 11;

export const LEGEND_ROW_HEIGHT = 14;

export const LEGEND_SWATCH_INSET = 2;

export const LEGEND_SWATCH_WIDTH = 12;

export const LEGEND_TEXT_OFFSET = 18;

export const LEGEND_ITEM_GAP = 12;

export const LEGEND_BASELINE_OFFSET = 8;

const FULLWIDTH =
  /[\u1100-\u115F\u2E80-\u303E\u3041-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uA000-\uA4CF\uAC00-\uD7A3\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFF60\uFFE0-\uFFE6]/u;

const NARROW_CHARS = new Set(" .,:;!|'`ijltfr()[]{}/\\-");
const WIDE_CHARS = new Set("mwMW@%");
const UPPER_OR_DIGIT = /[A-Z0-9$#]/u;

const EM_FULLWIDTH = 1;
const EM_NARROW = 0.34;
const EM_WIDE = 0.88;
const EM_UPPER_OR_DIGIT = 0.62;
const EM_DEFAULT = 0.53;

export function estimateTextWidth(
  text: string,
  fontSize: number = LEGEND_FONT_SIZE,
): number {
  let em = 0;
  for (const ch of text) {
    if (FULLWIDTH.test(ch)) em += EM_FULLWIDTH;
    else if (NARROW_CHARS.has(ch)) em += EM_NARROW;
    else if (WIDE_CHARS.has(ch)) em += EM_WIDE;
    else if (UPPER_OR_DIGIT.test(ch)) em += EM_UPPER_OR_DIGIT;
    else em += EM_DEFAULT;
  }
  return em * fontSize;
}

export interface LegendItemLayout {
  x: number;
  row: number;
}

export interface LegendLayout {
  items: LegendItemLayout[];
  rows: number;
  extraHeight: number;
}

export function layoutChartLegend(
  labels: readonly string[],
  availableWidth: number,
  fontSize: number = LEGEND_FONT_SIZE,
): LegendLayout {
  const items: LegendItemLayout[] = [];
  let row = 0;
  let cursor = 0;

  for (const label of labels) {
    const itemWidth = LEGEND_TEXT_OFFSET + estimateTextWidth(label, fontSize);
    if (cursor > 0 && cursor + itemWidth > availableWidth) {
      row += 1;
      cursor = 0;
    }
    items.push({ x: cursor, row });
    cursor += itemWidth + LEGEND_ITEM_GAP;
  }

  const rows = items.length === 0 ? 1 : row + 1;
  return { items, rows, extraHeight: (rows - 1) * LEGEND_ROW_HEIGHT };
}
