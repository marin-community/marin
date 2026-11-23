<script setup lang="ts">
/**
 * Renders free-form model output that may contain ``` fenced blocks: prose stays as normal
 * wrapped text, fenced segments render as monospace blocks (their language-tag line, if any, is
 * dropped). No markdown parsing beyond fence-splitting — this is enough to make code and math
 * traces readable without a markdown dependency. The splitting and drawing are the shared
 * Marina transcript renderer's.
 */
import { onMounted, ref } from 'vue'
import TranscriptView from '@marina/transcript.js'

const props = defineProps<{ text: string }>()

const container = ref<HTMLElement | null>(null)

onMounted(() => {
  if (container.value) TranscriptView.renderFenced(container.value, props.text)
})
</script>

<template>
  <div ref="container" class="fenced" />
</template>

<style scoped>
/* The shared renderer styles each segment; the wrapper only stacks them. */
.fenced { display: flex; flex-direction: column; gap: 0.5rem; }
</style>
