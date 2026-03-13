<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { controllerRpcCall } from '@/composables/useRpc'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { stateToName, stateDisplayName, statusColors } from '@/types/status'
import type {
  JobStatus, TaskStatus, LaunchJobRequest,
  GetJobStatusResponse, ListTasksResponse,
  ResourceUsage,
} from '@/types/rpc'
import { timestampMs, formatTimestamp, formatDuration, formatBytes, formatDeviceConfig } from '@/utils/formatting'
import PageShell from '@/components/layout/PageShell.vue'
import StatusBadge from '@/components/shared/StatusBadge.vue'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LogViewer from '@/components/shared/LogViewer.vue'

const router = useRouter()

const props = defineProps<{
  jobId: string
}>()

const TERMINAL_STATES = new Set(['succeeded', 'failed', 'killed', 'worker_failed', 'unschedulable'])

// -- State --

const job = ref<JobStatus | null>(null)
const jobRequest = ref<LaunchJobRequest | null>(null)
const tasks = ref<TaskStatus[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const profilingTaskId = ref<string | null>(null)

// -- Fetch --

async function fetchData() {
  error.value = null
  try {
    const [jobResp, tasksResp] = await Promise.all([
      controllerRpcCall<GetJobStatusResponse>('GetJobStatus', { jobId: props.jobId }),
      controllerRpcCall<ListTasksResponse>('ListTasks', { jobId: props.jobId }),
    ])
    if (!jobResp.job) {
      error.value = 'Job not found'
      return
    }
    job.value = jobResp.job
    jobRequest.value = jobResp.request ?? null
    tasks.value = tasksResp.tasks ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}


onMounted(fetchData)

// Auto-refresh while job is not terminal
const isTerminal = computed(() => {
  if (!job.value) return false
  return TERMINAL_STATES.has(stateToName(job.value.state))
})

const { stop: stopRefresh } = useAutoRefresh(fetchData, 10_000)

watch(isTerminal, (terminal) => {
  if (terminal) stopRefresh()
})

// -- Formatting helpers --

function jobDuration(j: JobStatus): string {
  const started = timestampMs(j.startedAt)
  if (!started) return '-'
  const ended = timestampMs(j.finishedAt) || Date.now()
  return formatDuration(started, ended)
}

function taskDuration(t: TaskStatus): string {
  const started = timestampMs(t.startedAt)
  if (!started) return '-'
  const ended = timestampMs(t.finishedAt) || Date.now()
  return formatDuration(started, ended)
}

function formatMemMb(usage: ResourceUsage | undefined): string {
  if (!usage?.memoryMb) return '-'
  const mb = parseInt(usage.memoryMb, 10)
  return `${mb} MB`
}

function formatCpu(usage: ResourceUsage | undefined): string {
  if (!usage || usage.cpuPercent === undefined || usage.cpuPercent === 0) return '-'
  return `${usage.cpuPercent.toFixed(0)}%`
}

function taskIndex(taskId: string): string {
  const last = taskId.split('/').pop()
  if (!last) return '-'
  const parsed = parseInt(last, 10)
  return isNaN(parsed) ? '-' : String(parsed)
}

// -- Computed --

const pageTitle = computed(() => {
  if (!job.value) return `Job: ${props.jobId}`
  const name = job.value.name
  return (name && name !== props.jobId) ? name : `Job: ${props.jobId}`
})

const subtitle = computed(() => {
  if (!job.value) return ''
  return (job.value.name && job.value.name !== props.jobId) ? `ID: ${props.jobId}` : ''
})

const taskCounts = computed(() => {
  const counts = { total: 0, succeeded: 0, running: 0, building: 0, assigned: 0, pending: 0, failed: 0 }
  for (const t of tasks.value) {
    counts.total++
    const state = stateToName(t.state)
    if (state === 'succeeded' || state === 'killed') counts.succeeded++
    else if (state === 'running') counts.running++
    else if (state === 'building') counts.building++
    else if (state === 'assigned') counts.assigned++
    else if (state === 'pending') counts.pending++
    else if (state === 'failed' || state === 'worker_failed') counts.failed++
  }
  return counts
})

const acceleratorDisplay = computed(() => {
  const j = job.value
  const req = jobRequest.value
  const base = formatDeviceConfig(j?.resources?.device)
    ?? formatDeviceConfig(req?.resources?.device)
  return base ?? '-'
})

const cpuDisplay = computed(() => {
  const mc = job.value?.resources?.cpuMillicores
  if (!mc) return '-'
  return String(mc / 1000)
})

const memoryDisplay = computed(() => {
  const mb = job.value?.resources?.memoryBytes
  if (!mb) return '-'
  return formatBytes(parseInt(mb, 10))
})

const diskDisplay = computed(() => {
  const db = job.value?.resources?.diskBytes
  if (!db) return '-'
  return formatBytes(parseInt(db, 10))
})

// -- Profiling --

function buildProfileType(profilerType: string, format: string | null): Record<string, unknown> {
  if (profilerType === 'cpu') return { cpu: { format: format ?? 'SPEEDSCOPE' } }
  if (profilerType === 'memory') return { memory: { format: format ?? 'FLAMEGRAPH' } }
  return { threads: {} }
}

function openThreadDump(taskId: string) {
  router.push(`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(taskId)}/threads`)
}

async function handleProfile(taskId: string, profilerType: string, format: string | null) {
  if (profilerType === 'threads') {
    openThreadDump(taskId)
    return
  }
  profilingTaskId.value = taskId
  try {
    const body = {
      target: taskId,
      durationSeconds: 10,
      profileType: buildProfileType(profilerType, format),
    }
    const resp = await controllerRpcCall<{ profileData?: string; error?: string }>('ProfileTask', body)
    if (resp.error) {
      alert(`${profilerType.toUpperCase()} profile failed: ${resp.error}`)
      return
    }
    if (resp.profileData) {
      const decoded = atob(resp.profileData)
      const blob = new Blob([decoded], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `profile-${taskId.replace(/\//g, '_')}.out`
      a.click()
      URL.revokeObjectURL(url)
    }
  } catch (e) {
    alert(`${profilerType.toUpperCase()} profile failed: ${e instanceof Error ? e.message : e}`)
  } finally {
    profilingTaskId.value = null
  }
}
</script>

<template>
  <PageShell :title="pageTitle" back-to="/" back-label="Jobs">
    <!-- Subtitle (job ID when name differs) -->
    <p v-if="subtitle" class="text-sm text-text-secondary font-mono -mt-4 mb-6">
      {{ subtitle }}
    </p>

    <!-- Loading -->
    <div v-if="loading" class="flex items-center justify-center py-12 text-text-muted text-sm">
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      Loading...
    </div>

    <!-- Error -->
    <div
      v-else-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Content -->
    <template v-else-if="job">
      <!-- Error banner -->
      <div
        v-if="job.error"
        class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
      >
        <span class="font-semibold">Error:</span> {{ job.error }}
      </div>

      <!-- Pending reason banner -->
      <div
        v-if="job.pendingReason"
        class="mb-4 px-4 py-3 bg-status-warning-bg border border-status-warning-border rounded-lg"
      >
        <span class="font-semibold text-status-warning text-sm">Scheduling Diagnostic:</span>
        <pre class="mt-2 p-3 bg-surface rounded text-xs font-mono whitespace-pre-wrap">{{ job.pendingReason }}</pre>
      </div>

      <!-- Info cards -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <InfoCard title="Job Status">
          <InfoRow label="State">
            <StatusBadge :status="job.state" size="sm" />
          </InfoRow>
          <InfoRow label="Started">
            <span class="font-mono">{{ formatTimestamp(job.startedAt) }}</span>
          </InfoRow>
          <InfoRow label="Finished">
            <span class="font-mono">{{ isTerminal ? formatTimestamp(job.finishedAt) : '-' }}</span>
          </InfoRow>
          <InfoRow label="Duration">
            <span class="font-mono">{{ jobDuration(job) }}</span>
          </InfoRow>
          <InfoRow label="Failures">
            {{ job.failureCount ?? 0 }}
          </InfoRow>
        </InfoCard>

        <InfoCard title="Task Summary">
          <InfoRow label="Total">{{ taskCounts.total }}</InfoRow>
          <InfoRow label="Completed">{{ taskCounts.succeeded }}</InfoRow>
          <InfoRow label="Running">{{ taskCounts.running }}</InfoRow>
          <InfoRow label="Building">{{ taskCounts.building }}</InfoRow>
          <InfoRow label="Assigned">{{ taskCounts.assigned }}</InfoRow>
          <InfoRow label="Pending">{{ taskCounts.pending }}</InfoRow>
          <InfoRow label="Failed">{{ taskCounts.failed }}</InfoRow>
        </InfoCard>

        <InfoCard title="Resources (per VM)">
          <InfoRow label="CPU">{{ cpuDisplay }}</InfoRow>
          <InfoRow label="Memory">{{ memoryDisplay }}</InfoRow>
          <InfoRow label="Disk">{{ diskDisplay }}</InfoRow>
          <InfoRow label="Accelerator">{{ acceleratorDisplay }}</InfoRow>
          <InfoRow label="Replicas">{{ tasks.length || '-' }}</InfoRow>
        </InfoCard>
      </div>

      <!-- Constraints -->
      <div
        v-if="jobRequest?.constraints && jobRequest.constraints.length > 0"
        class="mb-6 rounded-lg border border-surface-border bg-surface px-4 py-3"
      >
        <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
          Constraints
        </h3>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="(c, i) in jobRequest.constraints"
            :key="i"
            class="inline-block rounded bg-surface-sunken px-2 py-0.5 font-mono text-xs text-text-secondary"
          >
            {{ c.key }} {{ c.op }} {{ c.value?.stringValue ?? c.value?.intValue ?? '' }}
          </span>
        </div>
      </div>

      <!-- Tasks table -->
      <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary mb-3">
        Tasks
      </h3>

      <EmptyState v-if="tasks.length === 0" message="No tasks" />

      <div v-else class="overflow-x-auto">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Task</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">State</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Worker</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Mem</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">CPU</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Started</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Duration</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Exit</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Error</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Profiling</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="task in tasks"
              :key="task.taskId"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td class="px-3 py-2 text-[13px] font-mono">
                <RouterLink
                  :to="`/job/${encodeURIComponent(props.jobId)}/task/${encodeURIComponent(task.taskId)}`"
                  class="text-accent hover:underline"
                >
                  {{ taskIndex(task.taskId) }}
                </RouterLink>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <StatusBadge :status="task.state" size="sm" />
                <div v-if="task.pendingReason" class="text-xs text-status-warning mt-0.5 max-w-xs truncate">
                  {{ task.pendingReason }}
                </div>
              </td>
              <td class="px-3 py-2 text-[13px]">
                <RouterLink
                  v-if="task.workerId"
                  :to="'/worker/' + encodeURIComponent(task.workerId)"
                  class="text-accent hover:underline font-mono text-xs"
                >
                  {{ task.workerId }}
                </RouterLink>
                <span v-else class="text-text-muted">&mdash;</span>
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ formatMemMb(task.resourceUsage) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ formatCpu(task.resourceUsage) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ formatTimestamp(task.startedAt) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                {{ taskDuration(task) }}
              </td>
              <td class="px-3 py-2 text-[13px] font-mono">
                {{ TERMINAL_STATES.has(stateToName(task.state)) && task.exitCode !== undefined ? task.exitCode : '-' }}
              </td>
              <td class="px-3 py-2 text-xs text-text-muted max-w-xs truncate" :title="task.error ?? ''">
                {{ task.error || '-' }}
              </td>
              <td class="px-3 py-2 text-[13px]">
                <div v-if="stateToName(task.state) === 'running'" class="flex gap-1">
                  <button
                    class="px-2 py-0.5 text-[11px] font-semibold rounded bg-status-purple text-white hover:opacity-80 disabled:opacity-50"
                    :disabled="profilingTaskId === task.taskId"
                    @click="handleProfile(task.taskId, 'cpu', 'SPEEDSCOPE')"
                  >
                    {{ profilingTaskId === task.taskId ? '⏳' : 'CPU' }}
                  </button>
                  <button
                    class="px-2 py-0.5 text-[11px] font-semibold rounded bg-status-success text-white hover:opacity-80 disabled:opacity-50"
                    :disabled="profilingTaskId === task.taskId"
                    @click="handleProfile(task.taskId, 'memory', 'FLAMEGRAPH')"
                  >
                    {{ profilingTaskId === task.taskId ? '⏳' : 'MEM' }}
                  </button>
                  <button
                    class="px-2 py-0.5 text-[11px] font-semibold rounded bg-accent text-white hover:opacity-80 disabled:opacity-50"
                    :disabled="profilingTaskId === task.taskId"
                    @click="handleProfile(task.taskId, 'threads', null)"
                  >
                    {{ profilingTaskId === task.taskId ? '⏳' : 'THR' }}
                  </button>
                </div>
                <span v-else class="text-text-muted">&mdash;</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Job logs -->
      <div class="mt-6 mb-6">
        <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary mb-3">
          Job Logs
        </h3>
        <LogViewer :task-id="jobId" />
      </div>
    </template>

  </PageShell>
</template>
