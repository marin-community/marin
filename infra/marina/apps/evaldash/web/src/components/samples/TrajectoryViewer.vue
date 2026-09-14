<script setup lang="ts">
/**
 * Renders a parsed agentic trajectory (ATIF): the agent identity and run-level token totals, then
 * one card per step. The trajectory is fetched and parsed by the parent (AgenticSample); this
 * component is presentational.
 */
import { computed } from 'vue'
import type { Trajectory } from '@/types/api'
import TrajectoryStepCard from '@/components/samples/TrajectoryStepCard.vue'

const props = defineProps<{ trajectory: Trajectory }>()

const steps = computed(() => props.trajectory.steps ?? [])
const totals = computed(() => Object.entries(props.trajectory.final_metrics ?? {}))
const agent = computed(() => props.trajectory.agent ?? null)
</script>

<template>
  <div class="space-y-3">
    <div class="flex flex-wrap items-center gap-2 text-xs text-text-secondary">
      <span v-if="agent" class="font-mono">
        {{ agent.name }}<span v-if="agent.version" class="text-text-muted"> v{{ agent.version }}</span>
      </span>
      <span v-if="agent?.model_name" class="font-mono text-text-muted">{{ agent.model_name }}</span>
      <span class="tabular-nums text-text-muted">{{ steps.length }} steps</span>
      <span
        v-for="[name, value] in totals"
        :key="name"
        class="rounded border border-surface-border px-1.5 py-0.5 font-mono tabular-nums text-text-muted"
      >{{ name }} {{ value }}</span>
    </div>

    <div class="space-y-2">
      <TrajectoryStepCard v-for="step in steps" :key="step.step_id" :step="step" />
    </div>
  </div>
</template>
