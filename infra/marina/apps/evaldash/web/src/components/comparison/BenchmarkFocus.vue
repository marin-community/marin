<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import { INTERVAL_KIND, type ComparisonRow } from '@/types/api'
import { formatMeasurement, formatTimestamp, isPercentMetric } from '@/utils/formatting'
import { cellCoverageLabel, scoredItemsLabel, UNVERSIONED_COHORT } from '@/utils/panel'

const props = defineProps<{ row: ComparisonRow; models: string[] }>()
const cells = computed(() => Object.values(props.row.cells))
const commonMetric = computed(() => cells.value.every((cell) =>
  cell.metric === cells.value[0].metric && cell.metric_kind === cells.value[0].metric_kind,
) ? cells.value[0].metric : null)
const percent = computed(() => cells.value.every(isPercentMetric))
const fitted = ref(true)
const range = computed(() => {
  if (percent.value && !fitted.value) return { low: 0, high: 1, step: 0.25 }
  const low = Math.min(...cells.value.map((cell) => Math.min(cell.low, cell.value)))
  const high = Math.max(...cells.value.map((cell) => Math.max(cell.high, cell.value)))
  const pad = Math.max((high - low) * 0.12, percent.value ? 0.01 : Math.max(Math.abs(low) * 0.02, 0.001))
  const step = 10 ** Math.floor(Math.log10(high - low + 2 * pad)) / 2
  const lower = Math.floor((low - pad) / step) * step
  const upper = Math.ceil((high + pad) / step) * step
  return { low: percent.value ? Math.max(0, lower) : lower, high: percent.value ? Math.min(1, upper) : upper, step }
})
const axis = computed(() => {
  const { low, high, step } = range.value
  const stride = Math.ceil(Math.round((high - low) / step) / 4) * step
  const interior = [1, 2, 3].map((i) => low + i * stride).filter((tick) => tick <= high - stride / 2)
  const format = new Intl.NumberFormat(undefined, { maximumFractionDigits: percent.value ? 2 : 4 })
  return [low, ...interior, high].map((value) => ({ value, label: format.format(percent.value ? 100 * value : value) }))
})
const reference = computed(() => commonMetric.value && cells.value.length > 1 ? props.row.leader : null)
const models = computed(() => props.models.map((model) => {
  const cell = props.row.cells[model]
  const usePercent = cell && isPercentMetric(cell)
  return {
    model,
    cell,
    score: cell ? formatMeasurement(cell.value, usePercent) + (usePercent ? '%' : '') : '',
    interval: cell ? `${formatMeasurement(cell.low, usePercent)}–${formatMeasurement(cell.high, usePercent)}${usePercent ? '%' : ''}` : '',
    gap: gap(model),
  }
}))

function position(value: number) {
  return `${100 * (value - range.value.low) / (range.value.high - range.value.low)}%`
}

function signed(value: number) {
  return `${value > 0 ? '+' : ''}${formatMeasurement(value, percent.value)}`
}

function gap(model: string) {
  const interval = props.row.differences[model]
  if (!interval) return null
  const low = -interval.high
  const high = -interval.low
  return {
    estimate: signed(props.row.cells[model].value - props.row.cells[props.row.leader].value),
    interval: `${signed(low)} to ${signed(high)}`,
    evidence: low > 0 ? 'Higher than reference' : high < 0 ? 'Lower than reference' : 'Includes zero',
  }
}
</script>

<template>
  <section class="benchmark-focus" aria-label="Focused benchmark">
    <div class="focus-heading">
      <h2>{{ row.benchmark }}</h2>
      <p>{{ cells.length }} of {{ models.length }} models scored</p>
    </div>

    <figure>
      <figcaption v-if="commonMetric" class="axis-heading">
        <div>
          <p><strong>{{ commonMetric }}</strong> {{ percent ? '(%)' : '(raw score)' }}</p>
          <p>Point estimate and 95% interval</p>
        </div>
        <div v-if="percent" class="range-control" role="group" aria-label="Score axis">
          <button type="button" :aria-pressed="fitted" @click="fitted = true">Fit to scores</button>
          <button type="button" :aria-pressed="!fitted" @click="fitted = false">Full scale</button>
        </div>
      </figcaption>
      <figcaption v-else class="axis-heading">Recorded metrics differ. Each score is shown with its own 95% interval.</figcaption>

      <div v-if="commonMetric" class="figure-grid axis-grid" aria-hidden="true">
        <div class="plot-axis">
          <span v-for="(tick, index) in axis" :key="tick.value" :class="{ 'first-tick': index === 0, 'last-tick': index === axis.length - 1 }" :style="{ left: position(tick.value) }">
            {{ tick.label }}
          </span>
        </div>
      </div>

      <ul class="score-rows" aria-label="Model scores">
        <li v-for="{ model, cell, score, interval } in models" :key="model" class="figure-grid score-row">
          <ModelName class="focus-model-name" :model="model" />
          <div v-if="cell && commonMetric" class="plot-track" aria-hidden="true">
            <span class="plot-interval" :style="{ left: position(cell.low), right: `calc(100% - ${position(cell.high)})` }"></span>
            <span class="plot-point" :style="{ left: position(cell.value) }"></span>
          </div>
          <div v-if="cell" class="score-values">
            <RouterLink
              :to="{ name: 'run', params: { runId: cell.run_id } }"
              :aria-label="`${score}, open source run ${cell.run_id}`"
              class="score-link"
            >{{ score }}</RouterLink>
            <span><span class="sr-only">95% interval </span>{{ interval }}</span>
            <span v-if="!commonMetric">{{ cell.metric }}</span>
          </div>
          <p v-else class="score-values">No recorded score</p>
        </li>
      </ul>
    </figure>

    <section v-if="cells.length" class="focus-evidence" aria-label="Sources and differences">
      <h3>Sources and differences</h3>
      <ul>
        <li v-for="{ model, cell, gap } in models.filter((entry) => entry.cell)" :key="model" class="evidence-row">
          <ModelName class="focus-model-name" :model="model" />
          <template v-if="cell">
            <div>
              <p>
                <RouterLink
                  :to="{ name: 'run', params: { runId: cell.run_id } }"
                  :title="`Open source run ${cell.run_id}`"
                  class="score-link"
                >{{ cell.version ?? UNVERSIONED_COHORT }} <span aria-hidden="true">↗</span></RouterLink>
              </p>
              <p>{{ formatTimestamp(cell.created_at) }}</p>
              <p>{{ scoredItemsLabel(cell) }} scored · {{ cellCoverageLabel(cell) || 'Coverage unreported' }}</p>
              <p v-if="cell.interval_kind === INTERVAL_KIND.SAMPLING_ONLY">Sampling-only interval</p>
            </div>
            <p v-if="reference === model">Difference reference</p>
            <div v-else-if="reference && gap">
              <p><strong>{{ gap.estimate }}{{ percent ? ' pp' : '' }}</strong> vs reference</p>
              <p>95% difference interval {{ gap.interval }}{{ percent ? ' pp' : '' }}</p>
              <p>{{ gap.evidence }}</p>
            </div>
          </template>
        </li>
      </ul>
      <details v-if="reference">
        <summary>How differences are calculated</summary>
        <p>Each difference is this model’s score minus the reference score. The reference has the highest lower bound on its score interval. An interval including zero leaves the direction of the difference unresolved.</p>
      </details>
    </section>
  </section>
</template>

<style scoped>
.benchmark-focus {
  background: var(--c-surface);
  border: 1px solid var(--c-surface-border);
  border-radius: 0.75rem;
  overflow-wrap: anywhere;
  container-type: inline-size;
}

.focus-heading,
.axis-heading,
.axis-grid,
.score-row,
.focus-evidence {
  padding-inline: 1.15rem;
}

.focus-heading {
  padding-block: 0.78rem;
}

.focus-heading h2 {
  font-size: 1.35rem;
  font-weight: 650;
  line-height: 1.3;
}

.focus-heading p,
.axis-heading,
.score-values,
.focus-evidence {
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
}

.focus-heading p {
  margin-top: 0.2rem;
}

.axis-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.35rem 0.75rem;
  padding-block: 0.55rem 0.45rem;
  border-top: 1px solid var(--c-surface-border-subtle);
}

.range-control {
  display: flex;
  gap: 0.25rem;
}

.range-control button {
  min-height: 2rem;
  padding: 0.22rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 600;
}

.range-control button[aria-pressed='true'] {
  background: var(--c-accent-subtle);
  border-color: var(--c-accent-border);
  color: var(--c-accent);
}

.plot-axis {
  position: relative;
  min-height: 1.4rem;
  border-bottom: 1px solid var(--c-surface-border);
  color: var(--c-text-secondary);
  font-size: 0.74rem;
  font-variant-numeric: tabular-nums;
}

.plot-axis span {
  position: absolute;
  transform: translateX(-50%);
  white-space: nowrap;
}

.plot-axis .first-tick {
  transform: none;
}

.plot-axis .last-tick {
  transform: translateX(-100%);
}

.score-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.25rem 0.75rem;
  align-items: center;
  padding-block: 0.55rem;
}

.score-row + .score-row {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.focus-model-name {
  color: var(--c-text);
  font-size: 0.8rem;
  font-weight: 650;
}

.plot-track {
  position: relative;
  grid-column: 1 / -1;
  grid-row: 2;
  min-height: 0.75rem;
  background: var(--c-surface-sunken);
  border-radius: 0.15rem;
}

.plot-interval,
.plot-point {
  position: absolute;
  top: 50%;
}

.plot-interval {
  min-width: 2px;
  height: 0.25rem;
  transform: translateY(-50%);
  background: var(--c-accent);
}

.plot-point {
  width: 0.625rem;
  height: 0.625rem;
  transform: translate(-50%, -50%);
  border: 2px solid var(--c-surface);
  border-radius: 50%;
  background: var(--c-accent);
  box-shadow: 0 0 0 1px var(--c-accent);
}

.score-values {
  display: grid;
  line-height: 1.35;
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
}

.score-values .score-link {
  font-size: 0.9rem;
  font-weight: 650;
}

.score-link {
  color: var(--c-accent);
}

.score-link:hover {
  text-decoration: underline;
}

.focus-evidence {
  border-top: 1px solid var(--c-surface-border);
  padding-block: 0.75rem;
}

.focus-evidence h3 {
  color: var(--c-text);
  font-size: 0.9rem;
  font-weight: 650;
}

.evidence-row {
  display: grid;
  gap: 0.1rem 0.75rem;
  padding-block: 0.55rem;
}

.evidence-row + .evidence-row {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.focus-evidence details {
  margin-top: 0.35rem;
}

.focus-evidence details p {
  margin-top: 0.3rem;
  max-width: 75ch;
}

@container (min-width: 520px) {
  .figure-grid {
    display: grid;
    grid-template-columns: minmax(8rem, 1.4fr) minmax(0, 3fr) minmax(5.5rem, 1fr);
    gap: 0.75rem;
  }

  .axis-grid .plot-axis,
  .score-row .plot-track {
    grid-column: 2;
    grid-row: 1;
  }

  .score-row .score-values {
    grid-column: 3;
    grid-row: 1;
  }

  .evidence-row {
    grid-template-columns: minmax(8rem, 1.4fr) minmax(0, 1.7fr) minmax(0, 1.3fr);
  }
}
</style>
