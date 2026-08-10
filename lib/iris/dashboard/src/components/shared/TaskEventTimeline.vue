<script setup lang="ts">
import { formatRelativeTime } from '@/utils/formatting'

// A scheduling/lifecycle event decoded from the iris.task_event finelog
// namespace. Fields are all optional because proto3-JSON omits default scalars.
interface TaskEvent {
  ts?: number
  type?: string
  reason?: string
  message?: string
  source?: string
  count?: number
}

defineProps<{
  events: TaskEvent[]
}>()

function isWarning(type: string | undefined): boolean {
  return (type ?? '').toLowerCase() === 'warning'
}

// A Warning event gets the amber status palette; everything else stays neutral
// so a wall of Normal events doesn't read as a page full of alarms.
function chipClasses(type: string | undefined): string {
  const base =
    'inline-flex items-center rounded-full border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide'
  return isWarning(type)
    ? `${base} text-status-warning bg-status-warning-bg border-status-warning-border`
    : `${base} text-text-secondary bg-surface-sunken border-surface-border`
}

// The rail node echoes the chip: a filled amber dot for warnings, a hollow
// neutral dot otherwise.
function nodeClasses(type: string | undefined): string {
  return isWarning(type)
    ? 'bg-status-warning border-status-warning'
    : 'bg-surface border-surface-border'
}

function relativeTime(ts: number | undefined): string {
  return formatRelativeTime(Number(ts ?? 0))
}

function absoluteTime(ts: number | undefined): string {
  const ms = Number(ts ?? 0)
  return ms ? new Date(ms).toLocaleString() : ''
}
</script>

<template>
  <div v-if="events.length === 0" class="py-4 text-sm text-text-muted">
    No scheduling events recorded.
  </div>
  <ol v-else class="space-y-0">
    <li
      v-for="(ev, i) in events"
      :key="i"
      class="flex gap-3"
    >
      <!-- Rail: a severity-colored node with a connector to the next event.
           Order is meaningful here (newest first), so the timeline reads as a
           real sequence, not decoration. -->
      <div class="flex flex-col items-center">
        <span
          class="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full border"
          :class="nodeClasses(ev.type)"
        />
        <span
          v-if="i < events.length - 1"
          class="w-px flex-1 bg-surface-border"
        />
      </div>

      <!-- Content -->
      <div class="min-w-0 flex-1 pb-4">
        <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span
            class="font-mono text-xs text-text-muted tabular-nums"
            :title="absoluteTime(ev.ts)"
          >
            {{ relativeTime(ev.ts) }}
          </span>
          <span :class="chipClasses(ev.type)">{{ ev.type || 'Event' }}</span>
          <span class="font-mono text-[13px] font-semibold text-text break-all">
            {{ ev.reason }}
          </span>
          <span
            v-if="(ev.count ?? 0) > 1"
            class="font-mono text-xs text-text-muted tabular-nums"
            title="Times this event repeated"
          >
            ×{{ ev.count }}
          </span>
        </div>
        <p
          v-if="ev.message"
          class="mt-1 whitespace-pre-wrap break-words text-sm text-text-secondary"
        >
          {{ ev.message }}
        </p>
        <span
          v-if="ev.source"
          class="mt-1 inline-block rounded border border-surface-border-subtle
                 bg-surface-sunken px-1.5 py-0.5 font-mono text-[11px] text-text-muted"
        >
          {{ ev.source }}
        </span>
      </div>
    </li>
  </ol>
</template>
