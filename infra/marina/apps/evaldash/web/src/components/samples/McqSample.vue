<script setup lang="ts">
/**
 * Renders a loglikelihood multiple-choice sample: the shared few-shot context (collapsed by
 * default) followed by one row per choice showing its loglikelihood as a bar, with the model's
 * argmax pick and every correct choice called out when the source supplies them.
 */
import { computed } from 'vue'
import type { SampleRow } from '@/types/api'
import { goldChoiceIndices, sampleOutcome } from '@/utils/samples'
import PromptBlock from '@/components/samples/PromptBlock.vue'

const props = defineProps<{ sample: SampleRow }>()

const choices = computed(() => props.sample.choices ?? [])
const goldChoices = computed(() => new Set(goldChoiceIndices(props.sample)))
const isScored = computed(() => sampleOutcome(props.sample) === 'scored')

const logLikelihoods = computed(() =>
  choices.value.map((c) => c.loglikelihood).filter((v): v is number => v !== null),
)
const minLL = computed(() => Math.min(...logLikelihoods.value))
const maxLL = computed(() => Math.max(...logLikelihoods.value))

function barWidth(ll: number | null): number {
  if (ll === null || logLikelihoods.value.length === 0) return 8
  if (maxLL.value === minLL.value) return 100
  return 8 + ((ll - minLL.value) / (maxLL.value - minLL.value)) * 92
}

function rowClass(index: number): string {
  const isPick = index === props.sample.model_choice
  const isGold = goldChoices.value.has(index)
  if (isPick && isGold) return 'border-status-success-border bg-status-success-bg'
  if (isPick && isScored.value) return 'border-accent-border bg-accent-subtle'
  if (isPick) return 'border-status-danger-border bg-status-danger-bg'
  if (isGold) return 'border-status-success-border bg-status-success-bg'
  return 'border-surface-border'
}

function barClass(index: number): string {
  if (index === props.sample.model_choice) {
    if (isScored.value) return 'bg-accent'
    return goldChoices.value.has(index) ? 'bg-status-success' : 'bg-status-danger'
  }
  return 'bg-text-muted'
}
</script>

<template>
  <div class="space-y-4">
    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Prompt</h3>
      <PromptBlock :text="sample.prompt_text ?? ''" />
    </div>

    <div>
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-1">Choices</h3>
      <div class="space-y-1.5">
        <div v-for="(choice, i) in choices" :key="i" class="rounded border p-2.5" :class="rowClass(i)">
          <div class="flex items-start gap-2">
            <span class="inline-block rounded px-1.5 py-0.5 text-xs font-medium border border-surface-border bg-surface-raised font-mono">{{ choice.label }}</span>
            <span class="flex-1 text-sm whitespace-pre-wrap">{{ choice.text }}</span>
            <span class="font-mono text-xs text-text-secondary tabular-nums shrink-0">{{ choice.loglikelihood?.toFixed(2) ?? '—' }}</span>
            <span
              v-if="i === sample.model_choice"
              class="inline-block rounded px-1.5 py-0.5 text-xs border font-medium whitespace-nowrap shrink-0"
              :class="isScored
                ? 'bg-accent-subtle text-accent border-accent-border'
                : goldChoices.has(i)
                ? 'bg-status-success-bg text-status-success border-status-success-border'
                : 'bg-status-danger-bg text-status-danger border-status-danger-border'"
            >model pick</span>
            <span
              v-if="goldChoices.has(i)"
              class="inline-block rounded px-1.5 py-0.5 text-xs border font-medium whitespace-nowrap shrink-0 bg-status-success-bg text-status-success border-status-success-border"
            >target</span>
          </div>
          <div class="mt-1.5 h-1.5 rounded-full bg-surface-sunken overflow-hidden">
            <div class="h-full rounded-full" :class="barClass(i)" :style="{ width: `${barWidth(choice.loglikelihood)}%` }" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
