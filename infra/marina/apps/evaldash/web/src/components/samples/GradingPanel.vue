<script setup lang="ts">
/**
 * How one prediction was scored: the grader method, the headline metric and its extraction filter,
 * the score, a pass/fail chip, and the grader's verbatim `detail` JSON (collapsed). Rendered for
 * every sample kind so the browser shows why a sample is correct or incorrect.
 */
import { computed } from 'vue'
import type { SampleGrading } from '@/types/api'

const props = defineProps<{ grading: SampleGrading | null }>()

// The grader's raw output, pretty-printed. Empty `{}` carries nothing to show, so the detail block
// is suppressed for it; unparseable detail falls back to the raw string.
const prettyDetail = computed<string | null>(() => {
  const raw = props.grading?.detail
  if (!raw || raw.trim() === '{}' || raw.trim() === '') return null
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
})
</script>

<template>
  <section v-if="grading">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Grading</h3>
    <div class="rounded border border-surface-border bg-surface p-3 space-y-2">
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span class="rounded border border-surface-border bg-surface-sunken px-1.5 py-0.5 font-mono text-text-secondary">
          {{ grading.method }}
        </span>
        <span
          v-if="grading.metric"
          class="rounded border border-surface-border px-1.5 py-0.5 font-mono text-text-secondary"
        >{{ grading.metric }}</span>
        <span
          v-if="grading.filter"
          class="rounded border border-surface-border px-1.5 py-0.5 font-mono text-text-muted"
          title="extraction filter"
        >filter: {{ grading.filter }}</span>
        <span v-if="grading.score !== null" class="font-mono text-text-secondary tabular-nums">
          score {{ grading.score }}
        </span>
        <span
          v-if="grading.passed !== null"
          class="inline-block rounded px-1.5 py-0.5 border font-medium"
          :class="grading.passed
            ? 'bg-status-success-bg text-status-success border-status-success-border'
            : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
        >{{ grading.passed ? 'pass' : 'fail' }}</span>
      </div>

      <details v-if="prettyDetail">
        <summary class="text-xs text-text-muted cursor-pointer hover:text-text-secondary">Grader detail</summary>
        <pre class="mt-2 rounded border border-surface-border bg-surface-sunken p-3 text-[12px] font-mono overflow-auto max-h-72 whitespace-pre-wrap">{{ prettyDetail }}</pre>
      </details>
    </div>
  </section>
</template>
