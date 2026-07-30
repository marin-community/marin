<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatStderr } from '@/utils/formatting'
import type { LeaderboardEntry, Matrix } from '@/types/api'
import type { BestCell } from '@/components/charts/EvalRail.vue'
import EvalRail from '@/components/charts/EvalRail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

const router = useRouter()
const showArchived = ref(false)
const query = ref('')

const { data, refresh } = useApi<Matrix>(() =>
  showArchived.value ? 'api/matrix?include_archived=1' : 'api/matrix',
)

onMounted(refresh)
onViewRefresh(refresh)

const cellsByModel = computed<Record<string, Matrix['rows'][number]['cells']>>(() => {
  const out: Record<string, Matrix['rows'][number]['cells']> = {}
  for (const r of data.value?.rows ?? []) out[r.model] = r.cells
  return out
})

const best = computed<Record<string, BestCell>>(() => {
  const out: Record<string, BestCell> = {}
  for (const task of data.value?.tasks ?? []) {
    let bv = -Infinity
    let bm = ''
    for (const row of data.value?.rows ?? []) {
      const c = row.cells[task]
      if (c && c.value !== null && c.value > bv) {
        bv = c.value
        bm = row.model
      }
    }
    if (bm) out[task] = { value: bv, model: bm }
  }
  return out
})

const models = computed<LeaderboardEntry[]>(() => {
  const q = query.value.trim().toLowerCase()
  return [...(data.value?.leaderboard ?? [])]
    .filter((e) => !q || e.model.toLowerCase().includes(q))
    .sort((a, b) => Number(a.archived) - Number(b.archived))
})

function open(model: string) {
  router.push(`/models/${encodeURIComponent(model)}`)
}
</script>

<template>
  <section>
    <div class="mb-4">
      <h2 class="text-lg font-semibold">Models</h2>
      <p class="text-xs text-text-muted mt-0.5">Every model indexed. Open one for its full profile, cohorts, and runs.</p>
    </div>

    <div class="flex flex-wrap items-center gap-4 mb-4">
      <input
        v-model="query"
        type="search"
        placeholder="Filter models…"
        class="rounded border border-surface-border bg-surface px-3 py-1.5 text-sm font-mono min-w-[16rem]"
      />
      <label class="flex items-center gap-2 text-sm text-text-secondary">
        <input v-model="showArchived" type="checkbox" class="accent-accent" @change="refresh" />
        Show archived
      </label>
      <span class="text-xs text-text-muted ml-auto">{{ models.length }} models</span>
    </div>

    <EmptyState v-if="data && models.length === 0" icon="🔎" message="No models match." />

    <div v-else class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(320px, 1fr))">
      <button
        v-for="entry in models"
        :key="entry.model"
        class="text-left rounded-lg border border-surface-border bg-surface p-4 hover:border-accent-border hover:bg-surface-raised transition-colors"
        :class="{ 'opacity-50': entry.archived }"
        @click="open(entry.model)"
      >
        <div class="flex items-baseline gap-2 mb-1">
          <span class="font-mono font-semibold text-sm text-accent truncate">{{ entry.model }}</span>
          <span v-if="entry.version" class="rounded bg-surface-sunken px-1.5 py-0.5 text-[11px] font-mono text-text-muted">{{ entry.version }}</span>
        </div>
        <div class="flex items-baseline gap-2">
          <span class="font-mono text-2xl font-semibold tabular-nums">{{ formatScore(entry.score) }}</span>
          <span class="font-mono text-xs text-text-muted">{{ formatStderr(entry.score, entry.stderr) }}</span>
          <span
            class="ml-auto font-mono text-[11px] tabular-nums"
            :class="entry.covered <= 1 ? 'text-status-warning' : 'text-text-muted'"
          >{{ entry.covered }}/{{ entry.total }}<span v-if="entry.covered <= 1"> ⚠</span></span>
        </div>
        <div class="mt-3" @click.stop>
          <EvalRail
            :tasks="data?.tasks ?? []"
            :cells="cellsByModel[entry.model] ?? {}"
            :best="best"
            :model="entry.model"
            size="sm"
          />
        </div>
      </button>
    </div>
  </section>
</template>
