<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import type { ComparisonRow } from '@/types/api'
import { formatMeasurement, isPercentMetric } from '@/utils/formatting'
import ModelScoreSymbol from './ModelScoreSymbol.vue'

const props = defineProps<{
  rows: ComparisonRow[]
  models: string[]
  active?: string
  location: (benchmark: string) => { id: string; to: RouteLocationRaw }
}>()

const expandedModel = ref('')
const legend = computed(() => props.models.map(name => {
  // Preserve the tail of a clipped name so neighboring checkpoint IDs remain distinguishable.
  const tail = name.length > 16 ? name.slice(-8) : ''
  return { name, start: tail ? name.slice(0, -tail.length) : name, tail }
}))

const TICKS = [0, 25, 50, 75, 100]

const entries = computed(() => props.rows.map((row) => {
  const scores = props.models.flatMap((model, index) => {
    const cell = row.cells[model]
    if (!cell) return []
    const percent = isPercentMetric(cell)
    const value = formatMeasurement(cell.value, percent) + (percent ? '%' : '')
    return [{ index, cell, percent, value, label: `${model} scored ${value} (${cell.metric})` }]
  })
  const first = scores[0]?.cell
  const commonMetric = first && scores.every(({ cell }) => cell.metric === first.metric && cell.metric_kind === first.metric_kind)
  const plot = commonMetric && scores.every(({ percent }) => percent)
  const labels = props.models.map((model, index) => scores.find((score) => score.index === index)?.label ?? `${model} has no recorded score`)
  const description = [`${scores.length} of ${props.models.length} models scored`, ...labels].join('. ')
  return { ...props.location(row.benchmark), benchmark: row.benchmark, scores, metric: commonMetric ? first.metric : null, plot, description }
}))

const hasPercentPlots = computed(() => entries.value.some(({ plot }) => plot))
</script>

<template>
  <div>
    <ul class="model-legend" aria-label="Model score symbols">
      <li v-for="({ name, start, tail }, index) in legend" :key="name">
        <button type="button" class="model-key" :aria-label="name" :aria-expanded="expandedModel === name" :title="name"
          @click="expandedModel = expandedModel === name ? '' : name">
          <ModelScoreSymbol :index="index" />
          <span v-if="expandedModel === name" class="full-name">{{ name }}</span>
          <span v-else class="key-name"><span>{{ start }}</span><span>{{ tail }}</span></span>
        </button>
      </li>
    </ul>
    <div class="overview-grid overview-axis" aria-hidden="true">
      <span>Setting</span>
      <span v-if="hasPercentPlots" class="axis-ticks">
        <span v-for="tick in TICKS" :key="tick" :style="{ left: `${tick}%` }">{{ tick }}<template v-if="tick === 100">%</template></span>
      </span>
      <span v-else>Score</span>
      <span>Scored</span>
    </div>
    <ul class="benchmark-list" aria-label="Benchmark settings">
      <li v-for="entry in entries" :key="entry.benchmark">
        <RouterLink
          :id="entry.id"
          :to="entry.to"
          :aria-current="active === entry.benchmark ? 'true' : undefined"
          :aria-description="entry.description"
          class="overview-grid benchmark-link"
        >
          <span class="benchmark-label">{{ entry.benchmark }}</span>
          <span v-if="entry.plot" class="benchmark-track" aria-hidden="true">
            <span
              v-for="{ index, cell, label } in entry.scores"
              :key="index"
              class="track-marker"
              :style="{ left: `${100 * cell.value}%`, top: `${3 + index * 6}px` }"
              :title="label"
            >
              <ModelScoreSymbol :index="index" />
            </span>
          </span>
          <span v-else-if="entry.metric" class="raw-scores" aria-hidden="true">
            <span>{{ entry.metric }} (raw)</span>
            <span v-for="{ index, value, label } in entry.scores" :key="index" class="raw-score" :title="label">
              <ModelScoreSymbol :index="index" />
              {{ value }}
            </span>
          </span>
          <span v-else class="mixed-metrics" :title="entry.description" aria-hidden="true">Different metrics</span>
          <span class="score-coverage" aria-hidden="true">{{ entry.scores.length }}/{{ models.length }}</span>
          <span class="benchmark-chevron" aria-hidden="true">›</span>
        </RouterLink>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.model-legend {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.15rem 0.7rem;
  padding: 0.25rem 0.75rem 0.45rem;
  color: var(--c-text-secondary);
  font-size: 0.72rem;
}

.model-legend li { min-width: 0; }
.model-key {
  display: flex; align-items: baseline; gap: .4rem;
  max-width: 100%; padding: .08rem 0; border: 0; text-align: left;
  color: inherit; background: none; font: inherit;
}
.key-name { display: inline-flex; min-width: 0; border-bottom: 1px dotted var(--c-surface-border); }
.key-name > :first-child { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.key-name > :last-child { flex: none; }
.full-name { overflow-wrap: anywhere; }
.overview-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.7fr) 2.7rem 0.45rem;
  gap: 0.55rem;
  align-items: center;
  padding-inline: 0.75rem;
}

.overview-axis {
  min-height: 1.5rem;
  border-block: 1px solid var(--c-surface-border);
  color: var(--c-text-secondary);
  background: var(--c-surface-raised);
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}

.axis-ticks {
  position: relative;
  height: 1.15em;
}

.axis-ticks span {
  position: absolute;
  transform: translateX(-50%);
}

.axis-ticks span:first-child {
  transform: none;
}

.axis-ticks span:last-child {
  transform: translateX(-100%);
}

.benchmark-list li + li {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.benchmark-link {
  min-height: 2rem;
  padding-block: 0.22rem;
  color: var(--c-text);
  font-size: 0.78rem;
  overflow-wrap: anywhere;
}

.benchmark-link:hover,
.benchmark-link:focus-visible {
  background: var(--c-surface-raised);
}

.benchmark-link:hover .benchmark-label {
  text-decoration: underline;
  text-underline-offset: 0.16em;
}

.benchmark-link[aria-current='true'] {
  box-shadow: inset 3px 0 var(--c-accent);
  background: var(--c-accent-subtle);
  color: var(--c-accent);
}

.benchmark-link[aria-current='true'] .benchmark-label {
  font-weight: 650;
}

.benchmark-track {
  position: relative;
  display: block;
  height: 1.5rem;
  border-inline: 1px solid var(--c-surface-border);
  background: repeating-linear-gradient(
    to right,
    transparent 0 calc(25% - 1px),
    var(--c-surface-border) calc(25% - 1px) 25%
  );
}

.track-marker {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 0.8rem;
  height: 0.65rem;
  transform: translate(-50%, -50%);
}

.raw-scores {
  display: flex;
  flex-wrap: wrap;
  column-gap: 0.45rem;
  row-gap: 0.05rem;
}

.raw-score {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  white-space: nowrap;
}

.raw-scores,
.mixed-metrics,
.score-coverage {
  color: var(--c-text-secondary);
  font-size: 0.69rem;
  font-variant-numeric: tabular-nums;
}

.score-coverage {
  text-align: center;
}

.benchmark-chevron {
  color: var(--c-accent);
  font-size: 0.9rem;
}

@media (max-width: 480px) {
  .overview-grid {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.45fr) 1.85rem 0.35rem;
    gap: 0.4rem;
  }

  .benchmark-link {
    min-height: 2.25rem;
    font-size: 0.74rem;
  }

  .axis-ticks span:nth-child(even) {
    display: none;
  }
}
</style>
