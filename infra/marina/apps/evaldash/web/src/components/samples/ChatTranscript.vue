<script setup lang="ts">
/**
 * Renders a chat-formatted prompt (parsed from a JSON `[{role, content}, ...]` string) as a
 * sequence of role-labeled turn cards, left-border colored per role. The drawing is the shared
 * Marina transcript renderer; EvalDash maps its tokens onto it in `main.css`.
 */
import { onMounted, ref } from 'vue'
import type { ChatMessage } from '@/types/api'
import TranscriptView from '@marina/transcript.js'

const props = defineProps<{ messages: ChatMessage[] }>()

const container = ref<HTMLElement | null>(null)

onMounted(() => {
  if (container.value) TranscriptView.renderTranscript(container.value, props.messages)
})
</script>

<template>
  <div ref="container" />
</template>
