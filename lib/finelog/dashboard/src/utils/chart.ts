/**
 * Vocabulary shared between the chart and the controls that configure it.
 *
 * A Vue `<script setup>` block cannot export runtime values, so a constant the
 * chart and its pickers must agree on lives here rather than in the component.
 */

export type ChartMark = 'line' | 'bar' | 'scatter'

/**
 * Series a chart will draw. Above this a legend stops being readable and the
 * palette starts repeating colours, so a column with more distinct values is
 * not offered as a series split and the chart refuses to plot one.
 */
export const MAX_SERIES = 12
