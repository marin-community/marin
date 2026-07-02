<script setup lang="ts">
// A federation peer/cluster name that deep-links to that cluster's own dashboard
// when a URL is known, and falls back to plain text otherwise. Used wherever a
// federated job or task surfaces its owning cluster.
defineProps<{
  cluster: string
  href?: string
  title?: string
}>()
</script>

<template>
  <a
    v-if="href"
    :href="href"
    target="_blank"
    rel="noopener noreferrer"
    class="font-mono text-accent hover:underline"
    :title="title"
  >{{ cluster }} ↗</a>
  <span v-else class="font-mono">{{ cluster }}</span>
</template>
