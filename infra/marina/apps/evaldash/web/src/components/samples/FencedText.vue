<script setup lang="ts">
/**
 * Renders free-form model output that may contain ``` fenced blocks: prose stays as normal
 * wrapped text, fenced segments render as monospace blocks (their language-tag line, if any, is
 * dropped). No markdown parsing beyond fence-splitting — this is enough to make code and math
 * traces readable without a markdown dependency.
 */
import { computed } from 'vue'

const props = defineProps<{ text: string }>()

interface Segment {
  kind: 'prose' | 'code'
  text: string
}

function splitFences(text: string): Segment[] {
  const parts = text.split('```')
  const segments: Segment[] = []
  parts.forEach((part, i) => {
    if (i % 2 === 0) {
      if (part) segments.push({ kind: 'prose', text: part })
      return
    }
    const newlineIdx = part.indexOf('\n')
    const body = newlineIdx >= 0 ? part.slice(newlineIdx + 1) : part
    segments.push({ kind: 'code', text: body.replace(/\n$/, '') })
  })
  return segments
}

const segments = computed(() => splitFences(props.text))
</script>

<template>
  <div class="space-y-2">
    <template v-for="(seg, i) in segments" :key="i">
      <p v-if="seg.kind === 'prose'" class="whitespace-pre-wrap text-sm">{{ seg.text }}</p>
      <pre v-else class="rounded border border-surface-border bg-surface-sunken p-3 text-[13px] font-mono whitespace-pre-wrap overflow-auto">{{ seg.text }}</pre>
    </template>
  </div>
</template>
