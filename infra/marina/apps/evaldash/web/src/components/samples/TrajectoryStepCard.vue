<script setup lang="ts">
/**
 * One step of an agentic trajectory: the turn's source, the model message (fenced code split out,
 * long reasoning collapsed to a preview), any tool calls with their arguments, the resulting
 * observation output, and the step's token metrics.
 */
import { computed, ref } from 'vue'
import type { TrajectoryObservationResult, TrajectoryStep } from '@/types/api'
import FencedText from '@/components/samples/FencedText.vue'

const props = defineProps<{ step: TrajectoryStep }>()

const MESSAGE_PREVIEW = 600

const expanded = ref(false)

const message = computed(() => props.step.message ?? '')
const long = computed(() => message.value.length > MESSAGE_PREVIEW)
const shownMessage = computed(() =>
  long.value && !expanded.value ? `${message.value.slice(0, MESSAGE_PREVIEW)}…` : message.value,
)

const observationResults = computed(() => props.step.observation?.results ?? [])
const metricEntries = computed(() => Object.entries(props.step.metrics ?? {}))

// Terminal panes pad their output with runs of blank lines to a fixed height; collapse those runs
// and drop trailing whitespace so an observation shows its content instead of a screen of blank lines.
// A content-less result (a sub-agent delegation) falls back to its structured fields as JSON.
function observationText(result: TrajectoryObservationResult): string {
  if (typeof result.content !== 'string') return JSON.stringify(result, null, 2)
  return result.content.replace(/\n{3,}/g, '\n\n').replace(/\s+$/, '')
}

// A left-border accent per turn source, matching the chat transcript palette.
const SOURCE_BORDER: Record<string, string> = {
  user: 'border-l-text-muted',
  agent: 'border-l-status-success',
}
const borderClass = computed(() => SOURCE_BORDER[props.step.source] ?? 'border-l-surface-border')

function prettyArguments(args: Record<string, unknown>): string {
  return JSON.stringify(args, null, 2)
}
</script>

<template>
  <div class="rounded border border-surface-border bg-surface border-l-4 p-3 space-y-2" :class="borderClass">
    <div class="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-wider text-text-muted">
      <span class="font-semibold">Step {{ step.step_id }}</span>
      <span class="rounded bg-surface-sunken px-1 py-0.5 font-medium">{{ step.source }}</span>
      <span v-if="step.model_name" class="font-mono normal-case text-text-secondary">{{ step.model_name }}</span>
      <span
        v-for="[name, value] in metricEntries"
        :key="name"
        class="font-mono normal-case tabular-nums text-text-muted"
      >{{ name }} {{ value }}</span>
    </div>

    <div v-if="message">
      <FencedText :text="shownMessage" />
      <button
        v-if="long"
        class="mt-1 text-xs text-accent hover:text-accent-hover hover:underline"
        @click="expanded = !expanded"
      >{{ expanded ? 'Collapse' : `Show full message (${message.length} chars)` }}</button>
    </div>

    <div v-if="step.tool_calls?.length" class="space-y-1.5">
      <div
        v-for="(call, i) in step.tool_calls"
        :key="call.tool_call_id ?? i"
        class="rounded border border-surface-border bg-surface-sunken p-2"
      >
        <div class="text-[11px] font-mono text-accent">{{ call.function_name }}</div>
        <pre
          v-if="Object.keys(call.arguments ?? {}).length"
          class="mt-1 text-[12px] font-mono whitespace-pre-wrap overflow-auto max-h-48"
        >{{ prettyArguments(call.arguments) }}</pre>
      </div>
    </div>

    <div v-if="observationResults.length" class="space-y-1.5">
      <div v-for="(obs, i) in observationResults" :key="obs.source_call_id ?? i">
        <div class="text-[10px] uppercase tracking-wider text-text-muted mb-0.5">
          Observation<span v-if="obs.source_call_id" class="font-mono normal-case"> · {{ obs.source_call_id }}</span>
        </div>
        <pre class="rounded border border-surface-border bg-surface-sunken p-2 text-[12px] font-mono whitespace-pre-wrap overflow-auto max-h-64">{{ observationText(obs) }}</pre>
      </div>
    </div>
  </div>
</template>
