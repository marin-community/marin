<script setup lang="ts">
/** Render Harbor's retained SOTOPIA episode summary in conversation order. */
import { computed } from 'vue'
import type { SotopiaSummary } from '@/utils/sotopia'

const props = defineProps<{ summary: SotopiaSummary }>()

const messages = computed(() => props.summary.transcript.flatMap((turn, index) =>
  turn.map((message) => ({ ...message, turn: index + 1 })),
))
</script>

<template>
  <div class="space-y-4">
    <div class="flex flex-wrap gap-2 text-xs text-text-muted">
      <span>Environment <span class="font-mono">{{ summary.environment_id }}</span></span>
      <span>Combo <span class="font-mono">{{ summary.combo_id }}</span></span>
    </div>

    <section class="space-y-2">
      <h4 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Conversation</h4>
      <div
        v-for="(message, index) in messages"
        :key="index"
        class="rounded border border-surface-border bg-surface p-3 space-y-1"
      >
        <div class="text-xs text-text-muted">
          Turn {{ message.turn }} · <span class="font-semibold text-text-secondary">{{ message.sender }}</span>
          → {{ message.receiver }}
        </div>
        <pre class="text-sm whitespace-pre-wrap break-words font-sans text-text-primary">{{ message.message }}</pre>
      </div>
    </section>

    <section class="space-y-2">
      <h4 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Evaluator scores</h4>
      <div v-for="(score, index) in summary.scores" :key="index" class="rounded border border-surface-border bg-surface p-3">
        <div class="text-xs font-semibold text-text-secondary mb-2">
          {{ summary.agent_ids[index] ?? `Agent ${index + 1}` }}
          <span v-if="index === summary.evaluated_agent_index" class="text-accent"> · evaluated agent</span>
          <span v-if="summary.models[index]" class="font-mono font-normal text-text-muted"> · {{ summary.models[index] }}</span>
        </div>
        <dl class="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-xs">
          <template v-for="(value, name) in score" :key="name">
            <dt class="text-text-muted">{{ name }}</dt>
            <dd class="font-mono tabular-nums text-text-secondary">{{ value }}</dd>
          </template>
          <template v-if="summary.overall_scores[index] !== undefined">
            <dt class="font-semibold text-text-secondary">Overall</dt>
            <dd class="font-mono tabular-nums font-semibold text-text-secondary">{{ summary.overall_scores[index] }}</dd>
          </template>
        </dl>
      </div>
    </section>

    <section v-if="summary.reasoning" class="space-y-2">
      <h4 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Evaluator reasoning</h4>
      <pre class="rounded border border-surface-border bg-surface p-3 text-sm whitespace-pre-wrap break-words font-sans text-text-primary">{{ summary.reasoning }}</pre>
    </section>

    <section v-if="summary.context_budget" class="space-y-2">
      <h4 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Context budget</h4>
      <dl class="rounded border border-surface-border bg-surface p-3 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1 text-xs">
        <dt class="text-text-muted">Maximum input tokens</dt>
        <dd class="font-mono tabular-nums">{{ summary.context_budget.max_input_tokens }}</dd>
        <dt class="text-text-muted">Maximum prompt tokens</dt>
        <dd class="font-mono tabular-nums">{{ summary.context_budget.max_prompt_tokens }}</dd>
        <dt class="text-text-muted">Rejected requests</dt>
        <dd class="font-mono tabular-nums">{{ summary.context_budget.rejected_requests }}</dd>
      </dl>
    </section>
  </div>
</template>
