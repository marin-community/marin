<script setup lang="ts">
/**
 * One step of an agentic trajectory: the turn's source, the model message (fenced code split out,
 * long reasoning collapsed to a preview), any tool calls with their arguments, the resulting
 * observation output, and the step's token metrics. The drawing is the shared Marina transcript
 * renderer; EvalDash maps its tokens onto it in `main.css`.
 */
import { onMounted, ref } from 'vue'
import type { TrajectoryStep } from '@/types/api'
import TranscriptView from '@marina/transcript.js'

const props = defineProps<{ step: TrajectoryStep }>()

const container = ref<HTMLElement | null>(null)

onMounted(() => {
  if (container.value) TranscriptView.renderStepCard(container.value, props.step)
})
</script>

<template>
  <div ref="container" />
</template>
