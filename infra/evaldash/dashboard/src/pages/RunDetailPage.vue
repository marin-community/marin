<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import { formatScore, formatTimestamp, shortSha } from '@/utils/formatting'
import type { EvalRecord } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'
import JobsPanel from '@/components/runs/JobsPanel.vue'
import LogsPanel from '@/components/runs/LogsPanel.vue'
import SamplesPanel from '@/components/runs/SamplesPanel.vue'
import GroupLinks from '@/components/runs/GroupLinks.vue'

const props = defineProps<{ runId: string }>()

const { data, loading, error, refresh } = useApi<EvalRecord>(() => `api/runs/${props.runId}`)

onMounted(refresh)
watch(() => props.runId, refresh)
onViewRefresh(refresh)

interface MetricRow {
  task: string
  metric: string
  value: number
}

const metricRows = computed<MetricRow[]>(() => {
  const record = data.value
  if (!record) return []
  const rows: MetricRow[] = []
  for (const [task, metrics] of Object.entries(record.metrics ?? {})) {
    for (const [metric, value] of Object.entries(metrics)) {
      rows.push({ task, metric, value })
    }
  }
  return rows.sort((a, b) => a.task.localeCompare(b.task) || a.metric.localeCompare(b.metric))
})

const jobRoles = computed(() => Object.keys(data.value?.jobs ?? {}))

const copied = ref(false)
async function copyPath() {
  const record = data.value
  if (!record?.results_path) return
  try {
    await navigator.clipboard.writeText(record.results_path)
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch {
    // Clipboard denied (insecure context); the field is selectable regardless.
  }
}
</script>

<template>
  <section>
    <RouterLink to="/runs" class="text-xs text-accent hover:text-accent-hover hover:underline">← All runs</RouterLink>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2 my-4">
      {{ error }}
    </div>

    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <div v-else-if="data" class="mt-3 space-y-6">
      <div class="flex items-center gap-3 flex-wrap">
        <h2 class="text-lg font-semibold font-mono">{{ data.run_id }}</h2>
        <StatusChip :status="data.status" />
        <span
          v-if="data.version"
          class="rounded bg-surface-sunken px-1.5 py-0.5 text-xs font-mono text-text-secondary"
          :title="`version ${data.version}`"
        >{{ data.version }}</span>
        <span class="text-xs text-text-muted">{{ formatTimestamp(data.created_at) }}</span>
      </div>

      <p v-if="data.description" class="text-sm text-text-secondary -mt-3">{{ data.description }}</p>

      <div v-if="data.error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
        <span class="font-semibold">Error:</span> {{ data.error }}
      </div>

      <GroupLinks :run-id="data.run_id" />

      <!-- Summary grid -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="rounded-lg border border-surface-border bg-surface p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Model</h3>
          <dl class="text-sm space-y-1">
            <div class="flex gap-2"><dt class="text-text-muted w-24">name</dt><dd class="font-mono break-all">{{ data.model.name }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">location</dt><dd class="font-mono break-all">{{ data.model.location }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">backend</dt><dd>{{ data.model.backend }}</dd></div>
          </dl>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Eval</h3>
          <dl class="text-sm space-y-1">
            <div class="flex gap-2"><dt class="text-text-muted w-24">name</dt><dd>{{ data.eval.name }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">mechanism</dt><dd>{{ data.eval.mechanism }}</dd></div>
            <div class="flex gap-2">
              <dt class="text-text-muted w-24">tasks</dt>
              <dd>
                <span v-for="t in data.eval.tasks" :key="t.name" class="inline-block mr-2 font-mono text-[13px]">
                  {{ t.name }}<span v-if="t.num_fewshot != null" class="text-text-muted">/{{ t.num_fewshot }}shot</span>
                </span>
              </dd>
            </div>
          </dl>
        </div>
        <div class="rounded-lg border border-surface-border bg-surface p-4">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Hardware</h3>
          <dl class="text-sm space-y-1">
            <div class="flex gap-2"><dt class="text-text-muted w-24">platform</dt><dd>{{ data.hardware.platform }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">accelerator</dt><dd class="font-mono">{{ data.hardware.accelerator }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">region</dt><dd>{{ data.hardware.region_or_cluster }}</dd></div>
            <div class="flex gap-2"><dt class="text-text-muted w-24">user</dt><dd>{{ data.user }}</dd></div>
          </dl>
        </div>
      </div>

      <!-- Metrics -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Metrics</h3>
        <div v-if="metricRows.length" class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse text-sm">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-xs font-semibold uppercase tracking-wider text-text-secondary">
                <th class="px-3 py-2 text-left">Task</th>
                <th class="px-3 py-2 text-left">Metric</th>
                <th class="px-3 py-2 text-right">Value</th>
                <th class="px-3 py-2 text-right">Score</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="m in metricRows" :key="`${m.task}/${m.metric}`" class="border-b border-surface-border-subtle">
                <td class="px-3 py-2 font-mono text-[13px]">{{ m.task }}</td>
                <td class="px-3 py-2 font-mono text-[13px]">{{ m.metric }}</td>
                <td class="px-3 py-2 text-right tabular-nums text-text-secondary">{{ m.value }}</td>
                <td class="px-3 py-2 text-right tabular-nums">{{ formatScore(m.value) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-else class="text-sm text-text-muted">No metrics recorded.</p>
      </div>

      <!-- Samples (succeeded runs) -->
      <SamplesPanel v-if="data.status === 'succeeded'" :key="props.runId" :run-id="data.run_id" />

      <!-- Results path -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Results path</h3>
        <div class="flex items-center gap-2">
          <code class="flex-1 rounded border border-surface-border bg-surface-sunken px-3 py-2 text-[13px] font-mono break-all select-all">{{ data.results_path }}</code>
          <button
            class="text-xs px-2 py-2 rounded border border-surface-border hover:bg-surface-raised whitespace-nowrap"
            @click="copyPath"
          >
            {{ copied ? 'Copied' : 'Copy' }}
          </button>
        </div>
      </div>

      <!-- Provenance -->
      <div class="rounded-lg border border-surface-border bg-surface p-4">
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Provenance</h3>
        <dl class="text-sm grid grid-cols-1 md:grid-cols-3 gap-2">
          <div class="flex gap-2"><dt class="text-text-muted w-28">git sha</dt><dd class="font-mono" :title="data.provenance.git_sha">{{ shortSha(data.provenance.git_sha) }}</dd></div>
          <div class="flex gap-2"><dt class="text-text-muted w-28">image</dt><dd class="font-mono break-all">{{ data.provenance.evalchemy_image }}</dd></div>
          <div class="flex gap-2"><dt class="text-text-muted w-28">launch host</dt><dd class="font-mono break-all">{{ data.provenance.launch_host }}</dd></div>
        </dl>
      </div>

      <!-- Live iris job + attempt status for every role -->
      <JobsPanel :key="props.runId" :run-id="data.run_id" />

      <!-- Live finelog logs (collapsed), with the recorded log tails as fallback -->
      <LogsPanel :key="props.runId" :run-id="data.run_id" :roles="jobRoles" :log-tails="data.log_tails ?? {}" />
    </div>
  </section>
</template>
