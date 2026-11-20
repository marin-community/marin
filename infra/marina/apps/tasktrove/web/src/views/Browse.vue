<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { count, manifest, rowCount, tasks, type Manifest, type ParquetTask, type TaskFilters } from '../corpus'
import { taskPath } from '../routes'

const PAGE = 50
const route = useRoute()
const router = useRouter()
const dataset = ref<Manifest>()
const matches = ref<ParquetTask[]>([])
const total = ref(0)
const filteredTotal = ref(0)
const page = ref(0)
const problem = ref('')
const working = ref(false)
let generation = 0

const draft = reactive<TaskFilters>({ source: '', converter: '', mode: '', tag: '', environment: '', query: '' })

function value(name: keyof TaskFilters): string {
  const current = route.query[name]
  return typeof current === 'string' ? current : ''
}

function readRoute(): void {
  for (const name of Object.keys(draft) as (keyof TaskFilters)[]) draft[name] = value(name)
}

const filters = computed<TaskFilters>(() => ({
  source: value('source'),
  converter: value('converter'),
  mode: value('mode'),
  tag: value('tag'),
  environment: value('environment'),
  query: value('query'),
}))
const active = computed(() => Object.values(filters.value).some(Boolean))
const visible = computed(() => matches.value)
const sources = computed(() =>
  Object.entries(dataset.value?.by_source ?? {})
    .filter(([, statuses]) => (statuses.converted ?? 0) > 0)
    .sort((a, b) => a[0].localeCompare(b[0])),
)
const converters = computed(() => Object.keys(dataset.value?.by_converter ?? {}).sort())
const modes = computed(() => Object.keys(dataset.value?.by_mode ?? {}).sort())
const tags = computed(() => Object.keys(dataset.value?.by_tag ?? {}).sort())
const environments = computed(() => [...new Set(Object.values(dataset.value?.dockerfiles ?? {}).map((item) => item.base_image))].sort())
const firstMatch = computed(() => (visible.value.length ? page.value * PAGE + 1 : 0))
const lastMatch = computed(() => page.value * PAGE + visible.value.length)
const hasNext = computed(() => lastMatch.value < filteredTotal.value)

function applyFilters(): void {
  const query = Object.fromEntries(Object.entries(draft).filter(([, item]) => item.trim()))
  router.replace({ path: '/browse', query })
}

function resetFilters(): void {
  for (const name of Object.keys(draft) as (keyof TaskFilters)[]) draft[name] = ''
  applyFilters()
}

async function fill(targetPage: number, currentGeneration: number = generation): Promise<void> {
  if (working.value) return
  working.value = true
  problem.value = ''
  try {
    const found = await tasks(targetPage * PAGE, PAGE, filters.value)
    if (currentGeneration !== generation) return
    matches.value = found.rows
    filteredTotal.value = found.total
    page.value = targetPage
  } catch (error) {
    if (currentGeneration === generation) problem.value = String(error)
  } finally {
    if (currentGeneration === generation) working.value = false
  }
}

async function restart(): Promise<void> {
  generation += 1
  readRoute()
  matches.value = []
  page.value = 0
  filteredTotal.value = 0
  working.value = false
  await fill(0, generation)
}

function open(item: ParquetTask): void {
  router.push({ path: taskPath(item.row), query: { back: route.fullPath } })
}

onMounted(async () => {
  try {
    const [loadedManifest, loadedRows] = await Promise.all([manifest(), rowCount()])
    dataset.value = loadedManifest
    total.value = loadedRows
    await restart()
  } catch (error) {
    problem.value = String(error)
  }
})
watch(() => route.fullPath, restart)
</script>

<template>
  <header class="browse-heading">
    <div>
      <p class="eyebrow">Final output</p>
      <h1>Parquet viewer</h1>
      <p>
        Marina reads the final <code>tasks/part-00000.parquet</code> in S3 and returns paginated rows. Select a row
        to inspect the files in <code>task_binary</code>.
      </p>
    </div>
    <div class="coverage" v-if="dataset">
      <b>{{ count(total) }}</b><span>Parquet rows</span>
      <b>{{ modes.length }}</b><span>grader modes</span>
      <b>{{ environments.length }}</b><span>base environments</span>
    </div>
  </header>

  <form class="filters" @submit.prevent="applyFilters">
    <label>
      <span>Source</span>
      <select v-model="draft.source" @change="applyFilters" aria-label="Source">
        <option value="">All sources</option>
        <option v-for="[source, statuses] in sources" :key="source" :value="source">
          {{ source }} ({{ count(statuses.converted) }})
        </option>
      </select>
    </label>
    <label>
      <span>Grader</span>
      <select v-model="draft.mode" @change="applyFilters" aria-label="Grader">
        <option value="">All graders</option>
        <option v-for="mode in modes" :key="mode" :value="mode">{{ mode }} ({{ count(dataset?.by_mode[mode] ?? 0) }})</option>
      </select>
    </label>
    <label>
      <span>Converter</span>
      <select v-model="draft.converter" @change="applyFilters" aria-label="Converter">
        <option value="">All converters</option>
        <option v-for="converter in converters" :key="converter" :value="converter">{{ converter }}</option>
      </select>
    </label>
    <label>
      <span>Tag</span>
      <input v-model="draft.tag" list="tasktrove-tags" placeholder="Any tag" aria-label="Tag" @change="applyFilters" />
      <datalist id="tasktrove-tags"><option v-for="tag in tags" :key="tag" :value="tag" /></datalist>
    </label>
    <label>
      <span>Environment</span>
      <select v-model="draft.environment" @change="applyFilters" aria-label="Environment">
        <option value="">All environments</option>
        <option v-for="environment in environments" :key="environment" :value="environment">{{ environment }}</option>
      </select>
    </label>
    <label class="path-filter">
      <span>Path contains</span>
      <input v-model="draft.query" type="search" placeholder="repository, benchmark, or task id" aria-label="Path contains" />
    </label>
    <button type="submit">Apply</button>
    <button v-if="active" type="button" class="quiet" @click="resetFilters">Clear</button>
  </form>

  <p class="problem" v-if="problem">{{ problem }}</p>
  <div class="table-status" aria-live="polite">
    <span v-if="visible.length">
      Matches {{ count(firstMatch) }}–{{ count(lastMatch) }} of {{ count(filteredTotal) }}
    </span>
    <span v-else-if="working">Reading matching rows…</span>
    <span v-else>No matching rows.</span>
    <span v-if="working && visible.length">Reading ahead…</span>
  </div>

  <div class="data-table-wrap" v-if="visible.length">
    <table class="data-table task-table">
      <thead>
        <tr><th>Row</th><th>Task path</th><th>Source</th><th>Type</th><th>Grader</th><th>Environment</th><th>Tags</th></tr>
      </thead>
      <tbody>
        <tr
          v-for="item in visible"
          :key="item.row"
          role="link"
          tabindex="0"
          :aria-label="`Open ${item.path}`"
          @click="open(item)"
          @keydown.enter="open(item)"
        >
          <td class="row-number">{{ count(item.row) }}</td>
          <td><b>{{ item.path }}</b><small>{{ item.converter }}</small></td>
          <td>{{ item.source }}</td>
          <td>{{ item.family }}<small v-if="item.language">{{ item.language }}</small></td>
          <td><span class="mode">{{ item.mode }}</span></td>
          <td>{{ item.environment }}</td>
          <td><div class="tags"><span v-for="tag in item.tags.slice(0, 4)" :key="tag">{{ tag }}</span></div></td>
        </tr>
      </tbody>
    </table>
  </div>

  <nav class="pager" aria-label="Task pages" v-if="visible.length">
    <button type="button" :disabled="page === 0 || working" @click="fill(page - 1)">Previous</button>
    <span>Page {{ count(page + 1) }}</span>
    <button type="button" :disabled="!hasNext || working" @click="fill(page + 1)">Next</button>
  </nav>
</template>
