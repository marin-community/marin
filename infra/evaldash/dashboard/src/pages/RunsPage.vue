<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatStderr, formatTimestamp } from '@/utils/formatting'
import type { LaunchGroup, Meta, RunRow } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const route = useRoute()
const router = useRouter()

const filters = reactive({
  model: '',
  eval: '',
  user: '',
  status: '',
  limit: 200,
})

// The group filter is URL-driven (a chip), set from a run's group links; it is not a dropdown
// since group ids are opaque.
const group = computed(() => (typeof route.query.group === 'string' ? route.query.group : ''))

// "By launch" collapses runs into one row per serve group; "All runs" is the flat table. A group
// chip is a flat-mode concept, so arriving with one forces that view.
type View = 'launches' | 'runs'
const view = ref<View>(group.value ? 'runs' : 'launches')

function runsPath(): string {
  const params = new URLSearchParams()
  if (filters.model) params.set('model', filters.model)
  if (filters.eval) params.set('eval', filters.eval)
  if (filters.user) params.set('user', filters.user)
  if (filters.status) params.set('status', filters.status)
  if (group.value) params.set('group', group.value)
  params.set('limit', String(filters.limit))
  return `api/runs?${params.toString()}`
}

function groupsPath(): string {
  const params = new URLSearchParams()
  if (filters.model) params.set('model', filters.model)
  if (filters.user) params.set('user', filters.user)
  params.set('limit', String(filters.limit))
  return `api/groups?${params.toString()}`
}

const { data: meta, refresh: refreshMeta } = useApi<Meta>(() => 'api/meta')
const { data: runs, loading: runsLoading, error: runsError, refresh: refreshRuns } = useApi<RunRow[]>(runsPath)
const { data: groups, loading: groupsLoading, error: groupsError, refresh: refreshGroups } =
  useApi<LaunchGroup[]>(groupsPath)

const loading = computed(() => (view.value === 'launches' ? groupsLoading.value : runsLoading.value))
const error = computed(() => (view.value === 'launches' ? groupsError.value : runsError.value))

function refreshActive() {
  if (view.value === 'launches') refreshGroups()
  else refreshRuns()
}

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

// Re-query the active view whenever a filter or the view toggle changes.
watch(filters, refreshActive)
watch(view, refreshActive)

// A group chip only makes sense in flat mode, so selecting one switches views and refetches runs.
watch(group, (g) => {
  if (g) view.value = 'runs'
  refreshRuns()
})

// A manual or timed refresh reloads both the rows and the filter vocabulary.
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
  router.push({ path: '/runs', query: { group: groupId } })
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
        >By launch</button>
        <button
          class="px-3 py-1 rounded border"
          :class="view === 'runs'
            ? 'border-accent-border bg-accent-subtle text-accent'
            : 'border-surface-border text-text-muted hover:bg-surface-raised'"
          @click="view = 'runs'"
        >All runs</button>
      </div>
    </div>

    <!-- Filters -->
    <div class="flex flex-wrap gap-3 mb-4">
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Model
        <select v-model="filters.model" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[12rem]">
          <option value="">All</option>
          <option v-for="m in meta?.models ?? []" :key="m" :value="m">{{ m }}</option>
        </select>
      </label>
      <label v-if="view === 'runs'" class="flex flex-col text-xs text-text-secondary gap-1">
        Eval
        <select v-model="filters.eval" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[10rem]">
          <option value="">All</option>
          <option v-for="e in meta?.evals ?? []" :key="e" :value="e">{{ e }}</option>
        </select>
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        User
        <select v-model="filters.user" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[10rem]">
          <option value="">All</option>
          <option v-for="u in meta?.users ?? []" :key="u" :value="u">{{ u }}</option>
        </select>
      </label>
      <label v-if="view === 'runs'" class="flex flex-col text-xs text-text-secondary gap-1">
        Status
        <select v-model="filters.status" class="rounded border border-surface-border bg-surface px-2 py-1 text-sm min-w-[8rem]">
          <option value="">All</option>
          <option v-for="s in meta?.statuses ?? []" :key="s" :value="s">{{ s }}</option>
        </select>
      </label>
      <label class="flex flex-col text-xs text-text-secondary gap-1">
        Limit
        <input
          v-model.number="filters.limit"
          type="number"
          min="1"
          max="1000"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-sm w-24"
        />
      </label>
    </div>

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
        v-if="groups && groups.length === 0"
        icon="🔍"
        message="No launches match these filters."
      />
      <div v-else-if="groups" class="overflow-x-auto rounded-lg border border-surface-border">
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
            <template v-for="g in groups" :key="g.group_id">
              <tr
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors cursor-pointer"
                @click="toggleGroup(g.group_id)"
              >
                <td class="px-3 py-2 text-text-muted select-none">{{ expanded.has(g.group_id) ? '▾' : '▸' }}</td>
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
                          <template v-if="e.value !== null">
                            {{ formatScore(e.value) }}
                            <span class="text-text-muted text-xs">{{ formatStderr(e.value, e.stderr) }}</span>
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
      v-else-if="runs && runs.length === 0"
      icon="🔍"
      message="No runs match these filters."
    />

    <div v-else-if="runs" class="overflow-x-auto rounded-lg border border-surface-border">
      <table class="w-full border-collapse text-sm">
        <thead>
          <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
            <th class="px-3 py-2 text-left">Created</th>
            <th class="px-3 py-2 text-left">Model</th>
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
            v-for="row in runs"
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
