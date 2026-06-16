<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import type { SliceInfo } from '@/types/rpc'
import { SLICE_STATE_STYLES } from '@/types/status'
import {
  buildSliceView,
  sortSliceViews,
  type SliceJob,
  type SliceStatus,
  type SliceView,
} from '@/utils/slices'

const props = defineProps<{
  slices: SliceInfo[]
  /** worker_id → jobs running on that host (from the scheduler running buckets). */
  workerJobs: Map<string, SliceJob[]>
  /** Now, injected so relative ages match the rest of the page. */
  now: number
}>()

// A booting slice older than this is likely stuck provisioning — flag it.
const STUCK_BOOT_MS = 4 * 60 * 1000
// Collapse long slice lists; the actionable slices sort to the top.
const COLLAPSED_LIMIT = 12

const showAll = ref(false)

const views = computed<SliceView[]>(() =>
  sortSliceViews(props.slices.map((s) => buildSliceView(s, props.workerJobs, props.now))),
)

const visible = computed<SliceView[]>(() =>
  showAll.value ? views.value : views.value.slice(0, COLLAPSED_LIMIT),
)
const hiddenCount = computed(() => Math.max(0, views.value.length - COLLAPSED_LIMIT))

// -- Summary line: count by display status --

const STATUS_LABELS: Record<SliceStatus, string> = {
  in_use: 'in use',
  ready: 'ready',
  idle: 'idle',
  booting: 'booting',
  requesting: 'requesting',
  initializing: 'initializing',
  failed: 'failed',
}
const SUMMARY_ORDER: SliceStatus[] = ['in_use', 'ready', 'idle', 'booting', 'requesting', 'initializing', 'failed']

const summary = computed(() => {
  const counts: Partial<Record<SliceStatus, number>> = {}
  for (const v of views.value) counts[v.status] = (counts[v.status] ?? 0) + 1
  return SUMMARY_ORDER.filter((s) => counts[s]).map((s) => ({ status: s, label: STATUS_LABELS[s], count: counts[s]! }))
})

// -- Badge styling (reuse the canonical slice palette; add an idle style) --

interface BadgeStyle { label: string; bg: string; text: string; border: string; dot: string }

const IDLE_STYLE: BadgeStyle = {
  label: 'Idle (ready, no tasks — eligible for scale-down)',
  bg: 'bg-status-warning-bg',
  text: 'text-status-warning',
  border: 'border-status-warning-border',
  dot: 'bg-status-warning',
}

// Solid dot color per lifecycle key (literals so Tailwind includes them).
const DOT_CLASS: Record<string, string> = {
  ready: 'bg-status-success',
  in_use: 'bg-status-orange',
  requesting: 'bg-accent',
  booting: 'bg-status-purple',
  initializing: 'bg-status-warning',
  failed: 'bg-status-danger',
  idle: 'bg-status-warning',
}

function badge(status: SliceStatus): BadgeStyle {
  if (status === 'idle') return IDLE_STYLE
  const s = SLICE_STATE_STYLES[status]
  if (s) return { label: s.label, bg: s.bg, text: s.text, border: s.border, dot: DOT_CLASS[status] ?? 'bg-text-muted' }
  return { label: status, bg: 'bg-surface-sunken', text: 'text-text-muted', border: 'border-surface-border', dot: 'bg-text-muted' }
}

function isPending(v: SliceView): boolean {
  return v.status === 'booting' || v.status === 'requesting' || v.status === 'initializing'
}

// -- Formatting --

/** Trailing `YYYYMMDD-HHMM-hash` identifier; the group/region prefix is already row context. */
function shortSliceId(sliceId: string): string {
  const m = sliceId.match(/(\d{8}-\d{4}-[0-9a-f]+)$/)
  if (m) return m[1]
  const parts = sliceId.split('-')
  return parts.length > 3 ? parts.slice(-3).join('-') : sliceId
}

function formatAge(ms: number | null): string {
  if (ms == null) return '–'
  const s = Math.max(0, Math.floor(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ${m % 60}m`
  return `${Math.floor(h / 24)}d ${h % 24}h`
}
</script>

<template>
  <div class="space-y-2">
    <!-- Summary line -->
    <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
      <span class="font-semibold text-text-secondary">{{ views.length }} slice{{ views.length === 1 ? '' : 's' }}</span>
      <span v-for="s in summary" :key="s.status" class="inline-flex items-center gap-1">
        <span class="w-1.5 h-1.5 rounded-full" :class="badge(s.status).dot" />
        {{ s.count }} {{ s.label }}
      </span>
    </div>

    <!-- Slice rows -->
    <div class="divide-y divide-surface-border-subtle">
      <div
        v-for="v in visible"
        :key="v.sliceId"
        class="flex items-center gap-3 py-1.5 text-xs"
      >
        <!-- Status badge -->
        <span
          class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded border font-semibold flex-none w-[88px] justify-center"
          :class="[badge(v.status).bg, badge(v.status).text, badge(v.status).border]"
          :title="badge(v.status).label"
        >
          {{ STATUS_LABELS[v.status] }}
        </span>

        <!-- Slice id -->
        <span class="font-mono text-text-secondary flex-none w-[168px] truncate" :title="v.sliceId">
          {{ shortSliceId(v.sliceId) }}
        </span>

        <!-- Hosts + age -->
        <span class="text-text-muted flex-none w-[150px] tabular-nums">
          <template v-if="v.hostCount > 0">{{ v.hostCount }} host{{ v.hostCount === 1 ? '' : 's' }} ·</template>
          <span
            :class="isPending(v) && (v.ageMs ?? 0) > STUCK_BOOT_MS ? 'text-status-warning font-semibold' : ''"
            :title="isPending(v) && (v.ageMs ?? 0) > STUCK_BOOT_MS ? 'Booting longer than expected — possible stuck provisioning' : ''"
          >
            {{ isPending(v) ? 'booting' : 'up' }} {{ formatAge(v.ageMs) }}
          </span>
        </span>

        <!-- Context: jobs / idle / failure -->
        <span class="min-w-0 flex-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <!-- Failed: show the error -->
          <span v-if="v.status === 'failed'" class="text-status-danger truncate" :title="v.errorMessage">
            {{ v.errorMessage || 'slice failed' }}
          </span>

          <!-- Idle: how long it has been idle -->
          <span v-else-if="v.status === 'idle'" class="text-status-warning">
            idle {{ formatAge(v.idleForMs) }} — eligible for scale-down
          </span>

          <!-- Pending: waiting on cloud provisioning -->
          <span v-else-if="isPending(v)" class="text-text-muted italic">
            waiting for cloud capacity
          </span>

          <!-- In use: one chip per distinct job, deduped across hosts -->
          <template v-else-if="v.jobs.length > 0">
            <RouterLink
              v-for="job in v.jobs"
              :key="job.jobId"
              :to="`/job/${encodeURIComponent(job.jobId)}`"
              class="inline-flex items-center gap-1 text-accent hover:underline font-mono max-w-full truncate"
              :title="`${job.jobId} — ${job.userId || 'unknown'} · ${job.taskCount} task${job.taskCount === 1 ? '' : 's'} on ${job.hostCount}/${v.hostCount} host${v.hostCount === 1 ? '' : 's'}`"
            >
              {{ job.jobId }}
              <span v-if="job.userId" class="text-text-muted">· {{ job.userId }}</span>
              <span v-if="job.taskCount > 1" class="text-text-muted">×{{ job.taskCount }}</span>
            </RouterLink>
          </template>

          <!-- Ready but no job identity available (scheduler lag) -->
          <span v-else-if="v.taskCount > 0" class="text-text-muted">
            {{ v.taskCount }} task{{ v.taskCount === 1 ? '' : 's' }} running
          </span>
          <span v-else class="text-text-muted">available</span>
        </span>
      </div>
    </div>

    <!-- Show all toggle -->
    <button
      v-if="hiddenCount > 0"
      type="button"
      class="text-xs text-accent hover:underline cursor-pointer"
      @click="showAll = !showAll"
    >
      {{ showAll ? 'Show fewer' : `Show all ${views.length} slices (${hiddenCount} more)` }}
    </button>
  </div>
</template>
