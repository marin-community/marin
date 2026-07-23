<script setup lang="ts">
/**
 * Live iris job + attempt info for every role a run submitted. Fetched from
 * /api/runs/{id}/jobs, which resolves each role's job path against the iris controller over
 * Direct VPC egress. Outside the VPC the endpoint returns reachable=false per role, and the
 * panel still shows the role and its iris link.
 */
import { onMounted, watch } from 'vue'
import { useApi } from '@/composables/useApi'
import { formatDuration, formatMillis, protoTimestampMillis } from '@/utils/formatting'
import type { JobsResponse } from '@/types/api'

const props = defineProps<{ runId: string }>()

const { data, loading, error, refresh } = useApi<JobsResponse>(() => `api/runs/${props.runId}/jobs`)

onMounted(refresh)
watch(() => props.runId, refresh)

function irisJobUrl(path: string): string {
  return `https://iris.oa.dev/#/job/${encodeURIComponent(path)}`
}

function stateName(state: string): string {
  return state.replace(/^(JOB|TASK)_STATE_/, '').toLowerCase()
}

function taskIndex(taskId: string): string {
  return taskId.split('/').pop() ?? ''
}

// A terminal-state colour for a job/task/attempt state badge.
const OK = 'bg-status-success-bg text-status-success border-status-success-border'
const BAD = 'bg-status-danger-bg text-status-danger border-status-danger-border'
const WARN = 'bg-status-warning-bg text-status-warning border-status-warning-border'
const NEUTRAL = 'bg-surface-sunken text-text-secondary border-surface-border'
const STATE_BADGE: Record<string, string> = {
  succeeded: OK,
  running: 'bg-accent-subtle text-accent border-accent-border',
  pending: NEUTRAL,
  building: NEUTRAL,
  assigned: NEUTRAL,
  failed: BAD,
  worker_failed: WARN,
  preempted: WARN,
  killed: WARN,
  unschedulable: WARN,
  cosched_failed: WARN,
  missing: WARN,
}

function stateBadge(state: string): string {
  return STATE_BADGE[state] ?? NEUTRAL
}
</script>

<template>
  <div>
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">Jobs</h3>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
      {{ error }}
    </div>
    <div v-else-if="loading && !data" class="text-sm text-text-muted">Loading job status…</div>
    <p v-else-if="data && data.roles.length === 0" class="text-sm text-text-muted">
      No job links on this record — it predates job tracking (runs before 2026-07-19 02:27 UTC), or the
      orchestrator died before submitting any children.
    </p>

    <div v-else-if="data" class="space-y-4">
      <div v-for="role in data.roles" :key="role.role" class="rounded-lg border border-surface-border bg-surface p-4">
        <div class="flex items-center gap-2 flex-wrap mb-2">
          <span class="text-sm font-semibold">{{ role.role }}</span>
          <span
            v-if="role.job"
            class="inline-block rounded px-1.5 py-0.5 text-xs font-medium border whitespace-nowrap"
            :class="stateBadge(stateName(role.job.state))"
          >{{ stateName(role.job.state).replace(/_/g, ' ') }}</span>
          <span v-else-if="!role.reachable" class="text-xs text-text-muted" :title="role.error ?? ''">iris unreachable</span>
          <a
            :href="irisJobUrl(role.job_path)"
            target="_blank"
            rel="noopener"
            class="text-accent hover:text-accent-hover hover:underline text-xs whitespace-nowrap ml-auto"
          >iris ↗</a>
        </div>
        <code class="block font-mono text-[12px] break-all text-text-secondary mb-2 select-all">{{ role.job_path }}</code>

        <p v-if="role.job?.status_message" class="text-xs text-text-secondary mb-2">{{ role.job.status_message }}</p>
        <p v-if="role.job?.error" class="text-xs text-status-danger break-words mb-2">{{ role.job.error }}</p>

        <div v-if="role.tasks.length" class="overflow-x-auto rounded border border-surface-border">
          <table class="w-full border-collapse text-xs">
            <thead>
              <tr class="border-b border-surface-border bg-surface-raised text-text-secondary">
                <th class="px-2 py-1.5 text-left">Task</th>
                <th class="px-2 py-1.5 text-left">Att</th>
                <th class="px-2 py-1.5 text-left">State</th>
                <th class="px-2 py-1.5 text-left">Worker</th>
                <th class="px-2 py-1.5 text-right">Exit</th>
                <th class="px-2 py-1.5 text-left">Started</th>
                <th class="px-2 py-1.5 text-right">Dur</th>
                <th class="px-2 py-1.5 text-left">Error</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="task in role.tasks" :key="task.task_id">
                <tr
                  v-for="attempt in (task.attempts.length ? task.attempts : [null])"
                  :key="`${task.task_id}:${attempt?.attempt_id ?? 'x'}`"
                  class="border-b border-surface-border-subtle align-top"
                >
                  <td class="px-2 py-1.5 font-mono">{{ taskIndex(task.task_id) }}</td>
                  <td class="px-2 py-1.5 tabular-nums">{{ attempt ? attempt.attempt_id : task.current_attempt_id }}</td>
                  <td class="px-2 py-1.5">
                    <span
                      class="inline-block rounded px-1 py-0.5 border"
                      :class="stateBadge(stateName(attempt ? attempt.state : task.state))"
                    >{{ stateName(attempt ? attempt.state : task.state).replace(/_/g, ' ') }}</span>
                    <span v-if="attempt?.is_worker_failure" class="ml-1 text-status-warning" title="worker failure">⚑</span>
                  </td>
                  <td class="px-2 py-1.5 font-mono text-text-secondary">{{ (attempt ? attempt.worker_id : task.worker_id) || '—' }}</td>
                  <td class="px-2 py-1.5 text-right tabular-nums">{{ attempt ? attempt.exit_code : task.exit_code }}</td>
                  <td class="px-2 py-1.5 whitespace-nowrap text-text-secondary">{{ formatMillis(protoTimestampMillis(attempt ? attempt.started_at : task.started_at)) }}</td>
                  <td class="px-2 py-1.5 text-right whitespace-nowrap text-text-secondary">
                    {{ formatDuration(protoTimestampMillis(attempt ? attempt.started_at : task.started_at), protoTimestampMillis(attempt ? attempt.finished_at : task.finished_at)) }}
                  </td>
                  <td class="px-2 py-1.5 text-status-danger break-words max-w-[24ch]">{{ (attempt ? attempt.error : task.error) ?? '' }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
        <p v-else-if="role.reachable" class="text-xs text-text-muted">No tasks reported.</p>
      </div>
    </div>
  </div>
</template>
