<script setup lang="ts">
/**
 * Legend panel that documents every symbol, letter badge, colored dot, and
 * progress-bar color used across the dashboard. Rendered as a modal overlay
 * triggered from the app header's "?" button.
 *
 * The legend imports its style data from @/types/status so it stays in sync
 * with the source-of-truth definitions used by the other tabs — there is no
 * separate hardcoded list of colors or letters here.
 */
import { onMounted, onUnmounted } from 'vue'
import {
  SEGMENT_COLORS,
  SEGMENT_ORDER,
  SLICE_BADGE_ORDER,
  SLICE_STATE_STYLES,
  STATUS_COLOR_ORDER,
  stateDisplayName,
  statusColors,
} from '@/types/status'

const emit = defineEmits<{
  close: []
}>()

function close() {
  emit('close')
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') close()
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
})

// -- Availability badge examples --
// These mirror groupAvailabilityBadge() in AutoscalerTab.vue. Keep label text
// in sync with the logic there.
interface AvailabilityExample {
  label: string
  classes: string
  description: string
}

const AVAILABILITY_BADGES: AvailabilityExample[] = [
  {
    label: 'in-flight',
    classes: 'bg-status-purple-bg text-status-purple border-status-purple-border',
    description: 'Scale-up request is currently in flight to the provider',
  },
  {
    label: 'backoff',
    classes: 'bg-status-orange-bg text-status-orange border-status-orange-border',
    description: 'Group is backing off after a provisioning failure',
  },
  {
    label: 'quota exceeded',
    classes: 'bg-status-danger-bg text-status-danger border-status-danger-border',
    description: 'Provider quota is exhausted for this group',
  },
  {
    label: 'at capacity',
    classes: 'bg-status-warning-bg text-status-warning border-status-warning-border',
    description: 'Group has reached its configured maximum slice count',
  },
  {
    label: 'cooldown',
    classes: 'bg-accent-subtle text-accent border-accent-border',
    description: 'Scale-up is on cooldown after a recent action',
  },
  {
    label: 'tier-blocked',
    classes: 'bg-status-danger-bg text-status-danger border-status-danger-border opacity-60',
    description: 'Higher-tier group in the same pool is blocked; this tier will not be tried',
  },
]

// -- Action/state glyphs --
// These are scattered across components with no single registry; listing them
// here documents the conventions we use in templates.
interface GlyphExample {
  glyph: string
  meaning: string
}

const ACTION_GLYPHS: GlyphExample[] = [
  { glyph: '▶', meaning: 'Collapsed row — click to expand' },
  { glyph: '▼', meaning: 'Expanded row — click to collapse' },
  { glyph: '⟳', meaning: 'Auto-refresh is active' },
  { glyph: '↻', meaning: 'Manual refresh button' },
  { glyph: '⏳', meaning: 'Loading / waiting for data' },
  { glyph: '⏸', meaning: 'Paused or disabled feature' },
  { glyph: '⚠', meaning: 'Warning / failure count' },
  { glyph: '↕', meaning: 'Sortable column' },
]
</script>

<template>
  <div
    class="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 px-4 py-10"
    @click.self="close"
  >
    <div
      class="w-full max-w-3xl rounded-lg border border-surface-border bg-surface shadow-lg"
      role="dialog"
      aria-modal="true"
      aria-label="Dashboard legend"
    >
      <!-- Header -->
      <div class="flex items-center justify-between border-b border-surface-border px-5 py-3">
        <h2 class="text-base font-semibold text-text">Dashboard Legend</h2>
        <button
          class="text-text-muted hover:text-text transition-colors text-xl leading-none"
          aria-label="Close legend"
          @click="close"
        >
          &times;
        </button>
      </div>

      <div class="px-5 py-4 space-y-6 text-sm">
        <!-- Slice state badges -->
        <section>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            Slice Lifecycle Badges
          </h3>
          <p class="text-xs text-text-muted mb-3">
            Shown in the Autoscaler tab next to each scaling group. The number preceding
            each letter is the count of slices in that state.
          </p>
          <ul class="space-y-1.5">
            <li
              v-for="state in SLICE_BADGE_ORDER"
              :key="state"
              class="flex items-center gap-3"
            >
              <span
                :class="[
                  'inline-flex items-center justify-center w-9 px-1.5 py-0.5 rounded border text-xs font-semibold',
                  SLICE_STATE_STYLES[state].bg,
                  SLICE_STATE_STYLES[state].text,
                  SLICE_STATE_STYLES[state].border,
                ]"
              >
                N{{ SLICE_STATE_STYLES[state].letter }}
              </span>
              <span class="text-text">{{ SLICE_STATE_STYLES[state].label }}</span>
            </li>
            <li class="flex items-center gap-3">
              <span
                class="inline-flex items-center justify-center px-1.5 py-0.5 rounded border text-xs font-semibold
                       bg-status-warning-bg text-status-warning border-status-warning-border"
              >
                N idle
              </span>
              <span class="text-text">
                Ready slice that has been idle past its idle threshold and is
                a candidate for scale-down
              </span>
            </li>
          </ul>
        </section>

        <!-- Job / task status -->
        <section>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            Job &amp; Task Status
          </h3>
          <p class="text-xs text-text-muted mb-3">
            Colored dots and pill badges used on the Jobs tab and task lists.
          </p>
          <ul class="grid grid-cols-2 gap-y-1.5 gap-x-4">
            <li
              v-for="state in STATUS_COLOR_ORDER"
              :key="state"
              class="flex items-center gap-2"
            >
              <span
                :class="[
                  'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-semibold tracking-wide uppercase',
                  statusColors(state).text,
                  statusColors(state).bg,
                  statusColors(state).border,
                ]"
              >
                <span
                  :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', statusColors(state).dot]"
                />
                {{ stateDisplayName(state) }}
              </span>
            </li>
          </ul>
        </section>

        <!-- Progress bar segments -->
        <section>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            Progress Bar Segments
          </h3>
          <p class="text-xs text-text-muted mb-3">
            Stacked segments in each job's progress bar on the Jobs tab. Segment
            width is proportional to the number of tasks in that state.
          </p>
          <ul class="space-y-1.5">
            <li
              v-for="state in SEGMENT_ORDER"
              :key="state"
              class="flex items-center gap-3"
            >
              <span
                :class="['inline-block w-8 h-3 rounded-sm', SEGMENT_COLORS[state]]"
              />
              <span class="text-text">{{ stateDisplayName(state) }}</span>
            </li>
          </ul>
        </section>

        <!-- Autoscaler availability -->
        <section>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            Autoscaler Availability
          </h3>
          <p class="text-xs text-text-muted mb-3">
            Badges shown under a group name in the Waterfall Routing table when
            the group is not currently accepting new work.
          </p>
          <ul class="space-y-1.5">
            <li
              v-for="badge in AVAILABILITY_BADGES"
              :key="badge.label"
              class="flex items-start gap-3"
            >
              <span
                :class="[
                  'inline-flex items-center px-1.5 py-0.5 rounded border text-xs flex-shrink-0',
                  badge.classes,
                ]"
              >
                {{ badge.label }}
              </span>
              <span class="text-text-secondary">{{ badge.description }}</span>
            </li>
          </ul>
        </section>

        <!-- Action glyphs -->
        <section>
          <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
            Action &amp; State Glyphs
          </h3>
          <ul class="grid grid-cols-2 gap-y-1.5 gap-x-4">
            <li
              v-for="item in ACTION_GLYPHS"
              :key="item.glyph"
              class="flex items-center gap-3"
            >
              <span class="inline-flex items-center justify-center w-6 text-text text-base">
                {{ item.glyph }}
              </span>
              <span class="text-text-secondary text-xs">{{ item.meaning }}</span>
            </li>
          </ul>
        </section>
      </div>

      <!-- Footer -->
      <div class="border-t border-surface-border px-5 py-3 text-xs text-text-muted">
        Press <kbd class="px-1 py-0.5 rounded border border-surface-border bg-surface-sunken">Esc</kbd>
        or click outside to close.
      </div>
    </div>
  </div>
</template>
