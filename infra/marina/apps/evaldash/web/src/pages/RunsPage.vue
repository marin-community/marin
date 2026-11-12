<script setup lang="ts">
import { computed, onMounted, reactive, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatInterval, formatScore, formatTimestamp } from '@/utils/formatting'
import type { LaunchGroup, Meta, RunRow } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import FilterBar, { type Facet } from '@/components/shared/FilterBar.vue'

const route = useRoute()
const router = useRouter()

const DEFAULT_RESULT_LIMIT = 200
const MAX_RESULT_LIMIT = 1000

const FACETS = [
  { key: 'model', label: 'Model', searchable: true },
  { key: 'eval', label: 'Eval', searchable: true },
  { key: 'version', label: 'Version' },
  { key: 'status', label: 'Status' },
  { key: 'user', label: 'User' },
  { key: 'accelerator', label: 'Accelerator' },
]

function queryValue(key: string): string {
  const value = route.query[key]
  return typeof value === 'string' ? value : ''
}

function updateQuery(values: Record<string, string>) {
  const query = { ...route.query }
  for (const [key, value] of Object.entries(values)) {
    if (value) query[key] = value
    else delete query[key]
  }
  router.push({ path: '/runs', query })
}

const selected = computed<Record<string, string>>({
  get: () => Object.fromEntries(FACETS.map(({ key }) => [key, queryValue(key)])),
  set: (values) => updateQuery(Object.fromEntries(FACETS.map(({ key }) => [key, values[key] ?? '']))),
})

const limit = computed({
  get: () => {
    const value = Number(queryValue('limit'))
    return Number.isInteger(value) && value >= 1 && value <= MAX_RESULT_LIMIT ? value : DEFAULT_RESULT_LIMIT
  },
  set: (value: number) => {
    if (Number.isInteger(value) && value >= 1 && value <= MAX_RESULT_LIMIT) updateQuery({ limit: String(value) })
  },
})
const group = computed(() => queryValue('group'))
type View = 'launches' | 'runs'
const view = computed<View>({
  get: () => queryValue('view') === 'runs' || group.value ? 'runs' : 'launches',
  set: (value) => updateQuery({ view: value, ...(value === 'launches' ? { group: '' } : {}) }),
})

// Search the complete catalog before limiting results; options must include older and smoke runs.
function requestParams(): URLSearchParams {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(selected.value)) {
    if (value) params.set(key, value)
  }
  params.set('limit', String(limit.value))
  return params
}

function runsPath(): string {
  const params = requestParams()
  if (group.value) params.set('group', group.value)
  return `api/runs?${params}`
}

function groupsPath(): string {
  return `api/groups?${requestParams()}`
}

const { data: meta, error: metaError, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')
const { data: runs, loading: runsLoading, error: runsError, refresh: refreshRuns } = useApi<RunRow[]>(runsPath)
const { data: groups, loading: groupsLoading, error: groupsError, refresh: refreshGroups } =
  useApi<LaunchGroup[]>(groupsPath)
const loading = computed(() => view.value === 'launches' ? groupsLoading.value : runsLoading.value)
const error = computed(() => metaError.value || (view.value === 'launches' ? groupsError.value : runsError.value))

function refreshActive() {
  if (view.value === 'launches') refreshGroups()
  else refreshRuns()
}

const facets = computed<Facet[]>(() => FACETS.map((facet) => {
  const options = meta.value?.run_facets[facet.key] ?? []
  return {
    ...facet,
    options: facet.key === 'status' && view.value === 'launches'
      ? [...new Set([...options, 'mixed'])].sort()
      : options,
  }
}))
const runRows = computed(() => runs.value ?? [])
const launchRows = computed(() => groups.value ?? [])
const resultCount = computed(() => view.value === 'launches' ? launchRows.value.length : runRows.value.length)
const resultLabel = computed(() => {
  if (loading.value) return 'Loading…'
  const unit = view.value === 'launches' ? 'launch' : 'run'
  return `${resultCount.value} ${unit}${resultCount.value === 1 ? '' : 's'} shown`
})

// Expanded launches, keyed by group id.
const expanded = reactive(new Set<string>())
function toggleGroup(groupId: string) {
  if (expanded.has(groupId)) expanded.delete(groupId)
  else expanded.add(groupId)
}

onMounted(() => {
  refreshMeta()
  refreshActive()
})
watch(() => `${view.value}:${view.value === 'launches' ? groupsPath() : runsPath()}`, refreshActive)
onViewRefresh(() => {
  refreshMeta()
  refreshActive()
})

function tasksSummary(row: RunRow): string {
  if (!row.tasks || row.tasks.length === 0) return '—'
  if (row.tasks.length <= 3) return row.tasks.join(', ')
  return `${row.tasks.slice(0, 3).join(', ')} +${row.tasks.length - 3}`
}

function clearGroup() {
  const query = { ...route.query }
  delete query.group
  router.push({ path: '/runs', query })
}

function filterByGroup(groupId: string) {
  router.push({ path: '/runs', query: { ...route.query, group: groupId, view: 'runs' } })
}

function irisJobUrl(path: string): string {
  return `https://iris.oa.dev/#/job/${encodeURIComponent(path)}`
}

// Compact per-row job affordance: the serve/eval iris links, in that order.
function jobLinks(row: RunRow): { role: string; path: string }[] {
  const jobs = row.jobs ?? {}
  return ['serve', 'eval', 'orchestrator']
    .filter((role) => jobs[role])
    .map((role) => ({ role, path: jobs[role] }))
}
</script>

<template>
  <section>
    <div class="flex items-baseline justify-between mb-4">
      <h2 class="text-lg font-semibold">Runs</h2>
      <div class="flex items-center gap-1 text-sm">
        <button
          class="px-3 py-1 rounded border"
          :class="view === 'launches'
            ? 'border-accent-border bg-accent-subtle text-accent'
            : 'border-surface-border text-text-muted hover:bg-surface-raised'"
          @click="view = 'launches'"
          :aria-pressed="view === 'launches'"
        >By launch</button>
        <button
          class="px-3 py-1 rounded border"
          :class="view === 'runs'
            ? 'border-accent-border bg-accent-subtle text-accent'
            : 'border-surface-border text-text-muted hover:bg-surface-raised'"
          @click="view = 'runs'"
          :aria-pressed="view === 'runs'"
        >All runs</button>
      </div>
    </div>

    <FilterBar
      v-model="selected"
      :facets="facets"
      :result-label="resultLabel"
      class="mb-4"
    >
      <template #trailing>
        <label class="flex flex-col text-xs text-text-secondary gap-1">
          Result limit
          <input
            :value="limit"
            @change="limit = Number(($event.target as HTMLInputElement).value)"
            type="number"
            min="1"
            :max="MAX_RESULT_LIMIT"
            class="rounded border border-surface-border bg-surface px-2 py-1 text-sm w-24"
          />
        </label>
      </template>
    </FilterBar>

    <p class="mb-4 text-xs text-text-muted" role="status">
      Searches all recorded runs across cohorts. Showing up to {{ limit }} newest matches.
    </p>

    <!-- Active group chip (flat mode only) -->
    <div v-if="group && view === 'runs'" class="mb-4">
      <button
        class="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-full border border-accent-border bg-accent-subtle text-accent"
        title="Clear group filter"
        @click="clearGroup"
      >
        group: <span class="font-mono">{{ group }}</span> ✕
      </button>
    </div>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 mb-4">
      {{ error }}
    </div>

    <div
      v-if="loading && (view === 'launches' ? !groups : !runs)"
      class="text-sm text-text-muted py-12 text-center"
    >Loading…</div>

    <!-- By launch: one row per serve group, expandable to its evals -->
    <template v-else-if="view === 'launches'">
      <EmptyState
        v-if="launchRows.length === 0"
        icon="🔍"
        message="No launches match these filters."
      />
      <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse text-sm">
          <thead>
            <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
              <th class="px-3 py-2 text-left w-8"></th>
              <th class="px-3 py-2 text-left">Model</th>
              <th class="px-3 py-2 text-left">Version</th>
              <th class="px-3 py-2 text-left">Created</th>
              <th class="px-3 py-2 text-left">Status</th>
              <th class="px-3 py-2 text-left">Evals</th>
              <th class="px-3 py-2 text-left">Description</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="g in launchRows" :key="g.group_id">
              <tr
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
                @click="toggleGroup(g.group_id)"
              >
                <td class="px-3 py-2 text-text-muted">
                  <button
                    :aria-label="`Show evals for ${g.model_name}, ${formatTimestamp(g.created_at)}`"
                    :aria-expanded="expanded.has(g.group_id)"
                    class="px-1 py-1 hover:text-text"
                    @click.stop="toggleGroup(g.group_id)"
                  >{{ expanded.has(g.group_id) ? '▾' : '▸' }}</button>
                </td>
                <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">{{ g.model_name }}</td>
                <td class="px-3 py-2 whitespace-nowrap">
                  <span
                    v-if="g.version"
                    class="rounded bg-surface-sunken px-1.5 py-0.5 text-xs font-mono text-text-secondary"
                  >{{ g.version }}</span>
                  <span v-else class="text-text-muted">—</span>
                </td>
                <td class="px-3 py-2 whitespace-nowrap text-text-secondary">{{ formatTimestamp(g.created_at) }}</td>
                <td class="px-3 py-2"><StatusChip :status="g.status" /></td>
                <td class="px-3 py-2 tabular-nums text-text-secondary whitespace-nowrap">{{ g.n_succeeded }}/{{ g.n_evals }}</td>
                <td class="px-3 py-2 text-text-secondary max-w-[32ch] truncate" :title="g.description ?? ''">
                  {{ g.description ?? '—' }}
                </td>
              </tr>
              <tr v-if="expanded.has(g.group_id)" class="border-b border-surface-border-subtle bg-surface-sunken">
                <td></td>
                <td colspan="6" class="px-3 py-2">
                  <table class="w-full border-collapse text-sm">
                    <tbody>
                      <tr
                        v-for="e in g.evals"
                        :key="e.run_id"
                        class="border-b border-surface-border-subtle last:border-0"
                      >
                        <td class="px-3 py-1.5 whitespace-nowrap">{{ e.eval_name }}</td>
                        <td class="px-3 py-1.5"><StatusChip :status="e.status" /></td>
                        <td class="px-3 py-1.5 tabular-nums whitespace-nowrap">
                          <template v-if="e.headline">
                            {{ formatScore(e.headline.value) }}
                            <span class="text-text-muted text-xs">{{
                              formatInterval(e.headline.low, e.headline.high)
                            }}</span>
                          </template>
                          <span v-else class="text-text-muted">—</span>
                        </td>
                        <td class="px-3 py-1.5 text-right">
                          <RouterLink
                            :to="`/runs/${e.run_id}`"
                            class="text-accent hover:text-accent-hover hover:underline whitespace-nowrap"
                          >detail →</RouterLink>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </template>

    <EmptyState
      v-else-if="runRows.length === 0"
      icon="🔍"
      message="No runs match these filters."
    />

    <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
      <table class="w-full border-collapse text-sm">
        <thead>
          <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
            <th class="px-3 py-2 text-left">Created</th>
            <th class="px-3 py-2 text-left">Model</th>
            <th class="px-3 py-2 text-left">Version</th>
            <th class="px-3 py-2 text-left">Eval</th>
            <th class="px-3 py-2 text-left">Tasks</th>
            <th class="px-3 py-2 text-left">Status</th>
            <th class="px-3 py-2 text-left">User</th>
            <th class="px-3 py-2 text-left">Accelerator</th>
            <th class="px-3 py-2 text-left">Jobs</th>
            <th class="px-3 py-2 text-left"></th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in runRows"
            :key="row.run_id"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="px-3 py-2 whitespace-nowrap text-text-secondary">{{ formatTimestamp(row.created_at) }}</td>
            <td class="px-3 py-2 font-mono text-[13px] whitespace-nowrap">
              {{ row.model_name ?? '—' }}
              <button
                v-if="row.group_id && row.group_id !== row.run_id"
                class="ml-1 text-text-muted hover:text-accent"
                title="Filter to this run's serve group"
                @click="filterByGroup(row.group_id)"
              >⧉</button>
            </td>
            <td class="px-3 py-2 whitespace-nowrap">
              <span
                v-if="row.version"
                class="rounded bg-surface-sunken px-1.5 py-0.5 text-xs font-mono text-text-secondary"
              >{{ row.version }}</span>
              <span v-else class="text-text-muted">—</span>
            </td>
            <td class="px-3 py-2 whitespace-nowrap">{{ row.eval_name ?? '—' }}</td>
            <td class="px-3 py-2 text-text-secondary max-w-[24ch] truncate" :title="row.tasks?.join(', ')">
              {{ tasksSummary(row) }}
            </td>
            <td class="px-3 py-2"><StatusChip :status="row.status" /></td>
            <td class="px-3 py-2 whitespace-nowrap text-text-secondary">{{ row.user_name ?? '—' }}</td>
            <td class="px-3 py-2 whitespace-nowrap font-mono text-[13px] text-text-secondary">{{ row.accelerator ?? '—' }}</td>
            <td class="px-3 py-2 whitespace-nowrap">
              <span v-if="jobLinks(row).length === 0" class="text-text-muted">—</span>
              <a
                v-for="j in jobLinks(row)"
                :key="j.role"
                :href="irisJobUrl(j.path)"
                target="_blank"
                rel="noopener"
                class="mr-2 text-[11px] text-text-muted hover:text-accent hover:underline whitespace-nowrap"
                :title="j.path"
              >{{ j.role }}↗</a>
            </td>
            <td class="px-3 py-2 text-right">
              <RouterLink
                :to="`/runs/${row.run_id}`"
                class="text-accent hover:text-accent-hover hover:underline whitespace-nowrap"
              >
                detail →
              </RouterLink>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
