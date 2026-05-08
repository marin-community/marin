<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps<{
  content: string
}>()

const html = computed(() => marked.parse(props.content) as string)
</script>

<template>
  <div class="markdown-body" v-html="html" />
</template>

<style scoped>
.markdown-body :deep(p) {
  margin: 0.25rem 0;
}
.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0.25rem 0;
  padding-left: 1.25rem;
}
.markdown-body :deep(li) {
  margin: 0.1rem 0;
}
.markdown-body :deep(strong) {
  font-weight: 600;
}
.markdown-body :deep(em) {
  font-style: italic;
  color: var(--color-text-secondary);
}
.markdown-body :deep(code) {
  font-family: var(--font-mono);
  font-size: 0.8em;
  background: var(--color-surface-raised);
  border-radius: 3px;
  padding: 0.1em 0.3em;
}
.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3) {
  font-weight: 600;
  margin: 0.5rem 0 0.25rem;
}
</style>
