<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { onViewRefresh } from '@/composables/useRefresh'
import {
  formatDuration,
  formatScore,
  formatStderr,
  formatTimestamp,
  githubCommitUrl,
  objectStoreUrl,
  shortSha,
} from '@/utils/formatting'
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

// The eval's wall-clock duration, when the record captured a timing window. Null (rendered as "—")
// for records without timing.
const durationLabel = computed<string | null>(() => {
  const timing = data.value?.timing
  if (!timing) return null
  const start = new Date(timing.started_at).getTime()
  const end = timing.finished_at ? new Date(timing.finished_at).getTime() : null
  return formatDuration(start, end)
})

// When the run finished: the timing window's end, falling back to the record's created_at.
const finishedAt = computed(() => data.value?.timing?.finished_at ?? data.value?.created_at ?? null)

// Serving params as label/value pairs: the typed fields that were set, then the free-form extras.
// Empty when the run captured no serving params (older launches), which hides the section.
const servingRows = computed<{ label: string; value: string }[]>(() => {
  const s = data.value?.serving
  if (!s) return []
  const rows: { label: string; value: string }[] = []
  if (s.tensor_parallel_size != null) rows.push({ label: 'tensor parallel', value: String(s.tensor_parallel_size) })
  if (s.data_parallel_size != null) rows.push({ label: 'data parallel', value: String(s.data_parallel_size) })
  if (s.max_model_len != null) rows.push({ label: 'max model len', value: String(s.max_model_len) })
  if (s.max_gen_tokens != null) rows.push({ label: 'max gen tokens', value: String(s.max_gen_tokens) })
  for (const [key, value] of Object.entries(s.extra ?? {})) rows.push({ label: key, value })
  return rows
})

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

    <div v-else-if="data" class="mt-3 space-y-5">
      <!-- Identity: what this run is, at a glance -->
      <div class="space-y-1.5">
        <div class="flex items-center gap-3 flex-wrap">
          <h2 class="text-lg font-semibold font-mono">{{ data.run_id }}</h2>
          <StatusChip :status="data.status" />
          <span
            v-if="data.version"
            class="rounded bg-surface-sunken px-1.5 py-0.5 text-xs font-mono text-text-secondary"
            :title="`version ${data.version}`"
          >{{ data.version }}</span>
        </div>
        <div class="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-sm text-text-secondary">
          <span class="font-mono">{{ data.model.name }}</span>
          <span class="text-text-muted">·</span>
          <span>{{ data.eval.name }} <span class="text-text-muted">({{ data.eval.mechanism }})</span></span>
          <span class="text-text-muted">·</span>
          <span class="font-mono">{{ data.hardware.accelerator }}</span>
          <span class="text-text-muted">·</span>
          <span>{{ data.user }}</span>
        </div>
        <p v-if="data.description" class="text-sm text-text-secondary">{{ data.description }}</p>
      </div>

      <div v-if="data.error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
        <span class="font-semibold">Error:</span> {{ data.error }}
      </div>

      <!-- Result: the grade is the hero; duration and finish time sit beside it in one strip -->
      <div class="flex flex-wrap items-center gap-x-10 gap-y-3 rounded-lg border border-surface-border bg-surface px-5 py-4">
        <div>
          <div class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">Grade</div>
          <template v-if="data.headline">
            <div class="text-3xl font-semibold tabular-nums leading-none mt-1">
              {{ formatScore(data.headline.value) }}<span class="text-lg text-text-muted">%</span>
            </div>
            <div class="text-xs font-mono text-text-muted mt-1 truncate max-w-[16rem]" :title="data.headline.metric">
              {{ data.headline.metric }} {{ formatStderr(data.headline.value, data.headline.stderr) }}
            </div>
          </template>
          <div v-else class="text-3xl font-semibold text-text-muted leading-none mt-1">—</div>
        </div>
        <div>
          <div class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">Duration</div>
          <div class="text-lg font-semibold tabular-nums leading-none mt-1.5">{{ durationLabel ?? '—' }}</div>
        </div>
        <div>
          <div class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">Finished</div>
          <div class="text-sm font-medium mt-1.5">{{ formatTimestamp(finishedAt) }}</div>
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

      <!-- Per-question samples, when the run exported any. Rendered for every status: a failed eval
           can still have graded some questions, and the panel degrades gracefully when none exist. -->
      <SamplesPanel :key="props.runId" :run-id="data.run_id" />

      <!-- Reference detail below the results: how to reach sibling evals, the run's configuration,
           where its artifacts live, and its live job/log state. -->
      <GroupLinks :run-id="data.run_id" />

      <!-- Configuration: model, eval, hardware, and serving params in one card -->
      <div class="rounded-lg border border-surface-border bg-surface p-4">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4">
          <div>
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5">Model</h3>
            <dl class="text-sm space-y-1">
              <div class="flex gap-2"><dt class="text-text-muted w-24">name</dt><dd class="font-mono break-all">{{ data.model.name }}</dd></div>
              <div class="flex gap-2">
                <dt class="text-text-muted w-24">location</dt>
                <dd class="font-mono break-all">
                  <a
                    v-if="objectStoreUrl(data.model.location)"
                    :href="objectStoreUrl(data.model.location)!"
                    target="_blank"
                    rel="noopener"
                    class="text-accent hover:text-accent-hover hover:underline"
                  >{{ data.model.location }} ↗</a>
                  <span v-else>{{ data.model.location }}</span>
                </dd>
              </div>
              <div class="flex gap-2"><dt class="text-text-muted w-24">backend</dt><dd>{{ data.model.backend }}</dd></div>
            </dl>
          </div>
          <div>
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5">Eval</h3>
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
          <div>
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1.5">Hardware</h3>
            <dl class="text-sm space-y-1">
              <div class="flex gap-2"><dt class="text-text-muted w-24">platform</dt><dd>{{ data.hardware.platform }}</dd></div>
              <div class="flex gap-2"><dt class="text-text-muted w-24">accelerator</dt><dd class="font-mono">{{ data.hardware.accelerator }}</dd></div>
              <div class="flex gap-2"><dt class="text-text-muted w-24">region</dt><dd>{{ data.hardware.region_or_cluster }}</dd></div>
              <div class="flex gap-2"><dt class="text-text-muted w-24">user</dt><dd>{{ data.user }}</dd></div>
            </dl>
          </div>
        </div>
        <div v-if="servingRows.length" class="mt-4 pt-4 border-t border-surface-border-subtle">
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Serving</h3>
          <dl class="grid grid-cols-2 sm:grid-cols-4 gap-x-6 gap-y-1 text-sm">
            <div v-for="serving in servingRows" :key="serving.label" class="flex gap-2 min-w-0">
              <dt class="text-text-muted whitespace-nowrap">{{ serving.label }}</dt>
              <dd class="font-mono truncate" :title="serving.value">{{ serving.value }}</dd>
            </div>
          </dl>
        </div>
      </div>

      <!-- Results path -->
      <div>
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Results path</h3>
        <div class="flex items-center gap-2">
          <code class="flex-1 rounded border border-surface-border bg-surface-sunken px-3 py-2 text-[13px] font-mono break-all select-all">{{ data.results_path }}</code>
          <a
            v-if="objectStoreUrl(data.results_path)"
            :href="objectStoreUrl(data.results_path)!"
            target="_blank"
            rel="noopener"
            class="text-xs px-2 py-2 rounded border border-surface-border hover:bg-surface-raised whitespace-nowrap text-accent"
          >Open ↗</a>
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
          <div class="flex gap-2">
            <dt class="text-text-muted w-28">git sha</dt>
            <dd class="font-mono" :title="data.provenance.git_sha">
              <a
                :href="githubCommitUrl(data.provenance.git_sha)!"
                target="_blank"
                rel="noopener"
                class="text-accent hover:text-accent-hover hover:underline"
              >{{ shortSha(data.provenance.git_sha) }} ↗</a>
            </dd>
          </div>
          <div class="flex gap-2"><dt class="text-text-muted w-28">eval runtime</dt><dd class="font-mono break-all">{{ data.provenance.eval_runtime }}</dd></div>
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
