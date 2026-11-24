<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import SearchSelect from '@/components/shared/SearchSelect.vue'
import PolicyRejections from '@/components/shared/PolicyRejections.vue'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { isSmokeEval, MAX_COMPARE } from '@/constants'
import type { Meta, Panel } from '@/types/api'
import { formatMeasurement, formatMeasurementInterval, isPercentMetric } from '@/utils/formatting'
import { ALL_COHORTS, UNVERSIONED_COHORT, cellCoverageLabel, cohortWarning, scoredItemsLabel } from '@/utils/panel'
import { compareCells } from '@/utils/ranking'

const route = useRoute()
const router = useRouter()
const { data: meta, error: metaError, refresh: fetchMeta } = useApi<Meta>(() => 'api/meta')
const options = computed(() => meta.value?.evals.filter((name) => !isSmokeEval(name)) ?? [])
const benchmark = computed(() =>
  typeof route.query.benchmark === 'string' ? route.query.benchmark : '',
)
const cohort = computed(() =>
  typeof route.query.cohort === 'string' ? route.query.cohort : meta.value?.default_cohort ?? ALL_COHORTS,
)
function choose(key: 'cohort' | 'benchmark', value: string) {
  router.push({ query: { benchmark: benchmark.value, cohort: cohort.value, [key]: value } })
}

const suiteNames = computed(() => {
  const recorded = new Set(options.value)
  return (meta.value?.suites ?? []).map(({ suite, evals }) => ({
    name: suite,
    settings: evals.filter(setting => recorded.has(setting)),
  })).filter(suite => suite.settings.length)
})
const openSuites = ref<string[]>([])
function toggleSuite(name: string) {
  openSuites.value = openSuites.value.includes(name) ? openSuites.value.filter(suite => suite !== name) : [...openSuites.value, name]
}

const selected = ref<string[]>([])
const params = computed(() => new URLSearchParams({ benchmarks: benchmark.value, cohort: cohort.value }))
const { data, error, loading, refresh } = useApi<Panel>(() => 'api/panel?' + params.value)
const ready = computed(() => options.value.includes(benchmark.value))
watch([benchmark, cohort, ready], () => {
  selected.value = []
  if (ready.value) refresh()
}, { immediate: true })
onMounted(fetchMeta)
onViewRefresh(() => {
  fetchMeta()
  if (ready.value) refresh()
})

const warning = computed(() => meta.value && cohortWarning(cohort.value, meta.value))
const results = computed(() => (data.value?.rows ?? [])
  .flatMap(row => {
    const cell = row.cells[benchmark.value]
    return cell ? [{ model: row.model, cell }] : []
  })
  .sort((a, b) => compareCells(a.cell, b.cell) || a.model.localeCompare(b.model)))
const metrics = computed(() => [...new Set(results.value.map(({ cell }) => cell.metric))].join(', '))
const modelFilter = ref('')
const page = ref(0)
const pageSize = 20
const matching = computed(() => {
  const term = modelFilter.value.trim().toLowerCase()
  return term ? results.value.filter(({ model }) => model.toLowerCase().includes(term)) : results.value
})
const visible = computed(() => matching.value.slice(page.value * pageSize, (page.value + 1) * pageSize))
watch([benchmark, cohort, modelFilter], () => { page.value = 0 })
watch(() => matching.value.length, (length) => {
  page.value = Math.min(page.value, Math.max(0, Math.ceil(length / pageSize) - 1))
})
const gaps = computed(() => (data.value?.rows ?? [])
  .flatMap(row => {
    const gap = row.missing[benchmark.value]
    return gap ? [{ model: row.model, gap }] : []
  }))
const selectedModels = computed(() => selected.value.filter(model => results.value.some(result => result.model === model && result.cell.value !== 0)))
</script>

<template>
  <section class="benchmark-workspace">
    <RouterLink v-if="benchmark" :to="{ name: 'benchmarks', query: { cohort } }" class="library-back">← Benchmarks</RouterLink>
    <header class="workspace-heading">
      <div>
        <h1>{{ benchmark || 'Benchmarks' }}</h1>
        <p v-if="benchmark && data && ready && !loading" class="setting-meta">
          {{ results.length }} scored {{ results.length === 1 ? 'model' : 'models' }}<template v-if="metrics"> · {{ metrics }}</template>
        </p>
      </div>
      <RouterLink v-if="!benchmark" :to="{ name: 'panel', query: { cohort } }">Fleet coverage →</RouterLink>
      <label v-else class="cohort-choice">Cohort
        <select :value="cohort" @change="choose('cohort', ($event.target as HTMLSelectElement).value)">
          <option :value="ALL_COHORTS">All cohorts</option>
          <option v-for="version in meta?.versions" :key="version" :value="version">{{ version }}</option>
        </select>
      </label>
    </header>
    <div v-if="!benchmark" class="selectors">
      <SearchSelect label="Benchmark" :max-options="12" :options="options" :model-value="benchmark" @update:model-value="choose('benchmark', $event)" />
    </div>
    <p v-if="benchmark && warning" class="settings-note">{{ warning }}</p>
    <p v-if="metaError || error" role="alert">{{ metaError || error }}</p>
    <p v-else-if="!meta || (ready && loading)" role="status">Loading results…</p>
    <p v-else-if="!options.length">No benchmarks have been recorded.</p>
    <div v-else-if="!benchmark" class="benchmark-directory">
      <h2>Browse by suite</h2>
      <p class="directory-count">{{ options.length }} recorded settings</p>
      <section v-for="suite in suiteNames" :key="suite.name" class="directory-section">
        <header>
          <h3>{{ suite.name }}</h3>
          <span>{{ suite.settings.length }} {{ suite.settings.length === 1 ? 'setting' : 'settings' }}</span>
        </header>
        <div>
          <RouterLink v-for="setting in (openSuites.includes(suite.name) ? suite.settings : suite.settings.slice(0, 5))" :key="setting"
            :to="{ name: 'benchmarks', query: { benchmark: setting, cohort } }">{{ setting }}</RouterLink>
          <button v-if="suite.settings.length > 5" :aria-expanded="openSuites.includes(suite.name)" @click="toggleSuite(suite.name)">
            {{ openSuites.includes(suite.name) ? 'Show fewer' : 'Show all ' + suite.settings.length }}
          </button>
        </div>
      </section>
    </div>
    <p v-else-if="!ready">Unknown benchmark. Open the benchmark directory to choose a recorded setting.</p>
    <template v-else-if="data">
      <div class="results-toolbar">
        <RouterLink v-if="selectedModels.length >= 2" class="compare-action" :to="{ name: 'compare', query: { benchmarks: benchmark, cohort, models: selectedModels.join(',') } }">
          Compare {{ selectedModels.length }} models
        </RouterLink>
        <span v-else-if="results.length > 1" class="text-sm text-text-secondary">Select 2–{{ MAX_COMPARE }} models to compare</span>
        <div v-if="results.length > pageSize || modelFilter" class="model-tools">
          <input v-model="modelFilter" type="search" aria-label="Find model" placeholder="Find a model…" />
        <div>
          <button aria-label="Previous models" :disabled="page === 0" @click="page--">‹</button>
          <span>{{ matching.length ? page * pageSize + 1 : 0 }}–{{ page * pageSize + visible.length }} of {{ matching.length }} models</span>
          <button aria-label="Next models" :disabled="(page + 1) * pageSize >= matching.length" @click="page++">›</button>
        </div>
        </div>
      </div>
      <div v-if="visible.length" class="results-scroll">
        <table aria-label="Benchmark results">
          <thead><tr><th>Model</th><th>Score</th><th>95% interval</th><th>Graded items</th><th>Cohort</th><th>Source</th></tr></thead>
          <tbody>
            <tr v-for="{ model, cell } in visible" :key="model">
              <th scope="row">
                <div class="flex items-center gap-3">
                  <input type="checkbox" v-model="selected" :value="model" :aria-label="'Compare ' + model"
                    :disabled="cell.value === 0 || (selectedModels.length >= MAX_COMPARE && !selected.includes(model))" />
                  <RouterLink :to="{ name: 'model', params: { model }, query: { cohort } }">{{ model }}</RouterLink>
                </div>
              </th>
              <td data-label="Score" class="font-semibold">
                {{ formatMeasurement(cell.value, isPercentMetric(cell)) }}{{ isPercentMetric(cell) ? '%' : '' }}
                <small class="mobile-interval">95% CI {{ formatMeasurementInterval(cell.low, cell.high, isPercentMetric(cell)) }}</small>
              </td>
              <td data-label="95% interval" :title="cell.interval_kind">{{ formatMeasurementInterval(cell.low, cell.high, isPercentMetric(cell)) }}</td>
              <td data-label="Graded items">{{ scoredItemsLabel(cell) }}<small>{{ cellCoverageLabel(cell) || 'Coverage unknown' }}</small></td>
              <td data-label="Cohort">{{ cell.version || UNVERSIONED_COHORT }}</td>
              <td class="source-cell">
                <RouterLink :to="{ name: 'run', params: { runId: cell.run_id } }" :aria-label="'Open run ' + cell.run_id">Open run →</RouterLink>
                <details class="mobile-evidence">
                  <summary :aria-label="'Measurement details for ' + model">Info</summary>
                  <div>{{ cell.version || UNVERSIONED_COHORT }} · {{ scoredItemsLabel(cell) }} · {{ cellCoverageLabel(cell) || 'Coverage unknown' }}</div>
                  <RouterLink :to="{ name: 'run', params: { runId: cell.run_id } }">Open source run →</RouterLink>
                </details>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else>{{ results.length ? 'No models match this search.' : 'No scored models for this benchmark and cohort.' }}</p>
      <details v-if="gaps.length" class="mt-4">
        <summary>Runs without an admitted score ({{ gaps.length }})</summary>
        <ul class="mt-2 space-y-2">
          <li v-for="{ model, gap } in gaps" :key="model">
            <RouterLink :to="{ name: 'run', params: { runId: gap.run_id } }">{{ model }}</RouterLink> · {{ gap.reason }}
          </li>
        </ul>
      </details>
      <PolicyRejections :rejections="data.policy_rejections.filter(row => row.benchmark === benchmark)" scope="this benchmark and cohort" class="mt-3" />
    </template>
  </section>
</template>

<style scoped>
.mobile-evidence, td .mobile-interval { display: none; }
.library-back { display: inline-block; font-size: .85rem; margin-bottom: .5rem; }
.setting-meta { margin-top: .4rem; color: var(--c-text-secondary); font-size: .9rem; }
.cohort-choice { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; font-size: .82rem; }
.cohort-choice select { width: min(16rem, 100%); min-width: 0; }
.benchmark-directory { margin-top: 1.25rem; max-width: 68rem; }
.directory-count, .directory-section header span { color: var(--c-text-secondary); font-size: .83rem; }
.directory-section { display: grid; grid-template-columns: minmax(8rem, 1fr) 3fr; gap: 1.25rem; padding: .78rem 0; border-top: 1px solid var(--c-surface-border); }
.directory-section:first-of-type { margin-top: .8rem; }
.directory-section h3 { font-weight: 650; font-size: 1.05rem; }
.directory-section > div { display: flex; flex-wrap: wrap; align-items: baseline; gap: .65rem 1.1rem; font-size: .9rem; }
.directory-section a { overflow-wrap: anywhere; }
.directory-section button { color: var(--c-text-secondary); text-decoration: underline; font-size: .8rem; }
@media (max-width: 35rem) { .directory-section { grid-template-columns: 1fr; gap: .5rem; } }
.benchmark-workspace { color: var(--c-text); }
.workspace-heading, .results-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 1rem; }
.workspace-heading { margin: 0 0 .65rem; }

a { color: var(--c-accent); }
a:hover { text-decoration: underline; }
.selectors { display: grid; grid-template-columns: minmax(14rem, 28rem) minmax(14rem, 20rem); gap: 1rem; }
.settings-note { margin-top: .9rem; color: var(--c-text-secondary); border-left: 3px solid var(--c-status-warning); padding: .2rem .7rem; font-size: .84rem; }
.results-toolbar { gap: .4rem 1rem; margin: .65rem 0; }
h2 { overflow-wrap: anywhere; font-size: 1.3rem; font-weight: 650; }
td small { color: var(--c-text-secondary); font-size: .8rem; }
.compare-action { background: var(--c-accent); color: var(--c-surface); border-radius: .4rem; padding: .55rem .8rem; }
.model-tools, .model-tools > div { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .5rem; }
.model-tools { font-size: .85rem; }
.results-scroll { border: 1px solid var(--c-surface-border); border-radius: .6rem; overflow: auto; background: var(--c-surface); }
table { width: 100%; border-collapse: collapse; text-align: left; font-size: .9rem; font-variant-numeric: tabular-nums; }
td, th { padding: .42rem .72rem; border-bottom: 1px solid var(--c-surface-border-subtle); }
thead { background: var(--c-surface-raised); color: var(--c-text-secondary); font-size: .8rem; }
tbody th { max-width: 24rem; overflow-wrap: anywhere; }
td { white-space: nowrap; }
small { display: block; }
@media (max-width: 42rem) {
  .selectors { grid-template-columns: minmax(0, 1fr); }
  .results-scroll table, .results-scroll tbody { display: block; }
  thead { position: absolute; clip-path: inset(50%); height: 1px; width: 1px; overflow: hidden; }
  .workspace-heading { align-items: center; }
  .cohort-choice { max-width: 48%; gap: .25rem; }
  .cohort-choice select { width: 100%; }
  tbody tr { display: grid; grid-template-columns: minmax(0, 1fr) auto 2.8rem; gap: .4rem; align-items: center; padding: .35rem .45rem; }
  tbody tr + tr { border-top: 1px solid var(--c-surface-border); }
  tbody th, td { padding: 0; border: 0; white-space: normal; overflow-wrap: anywhere; }
  td[data-label="Score"] { text-align: right; }
  td .mobile-interval { display: block; font-weight: 400; font-size: .7rem; }
  td[data-label="95% interval"], td[data-label="Graded items"], td[data-label="Cohort"], .source-cell > a { display: none; }
  .mobile-evidence { display: block; min-width: 0; color: var(--c-text-secondary); font-size: .8rem; }
  .mobile-evidence summary { cursor: pointer; align-content: center; }
  .source-cell:has(details[open]) { grid-column: 1 / -1; }
  .model-tools { flex-wrap: nowrap; width: 100%; }
  .model-tools input { width: 9rem; min-width: 4rem; flex: 1; }
  .model-tools > div { flex-wrap: nowrap; }
}
</style>
