<script setup lang="ts">
// Shown in place of the log viewer for a federated job or task: its logs are
// written on the peer cluster, not the local log store, so the local proxy has
// nothing for it. Links to the peer's own dashboard, where the entity renders
// natively with working logs.
defineProps<{
  cluster: string
  subject: string
  href?: string
}>()
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface px-4 py-3 text-sm text-text-secondary">
    This {{ subject }} runs on the <span class="font-mono text-text">{{ cluster }}</span> cluster;
    its logs live there.
    <a
      v-if="href"
      :href="href"
      target="_blank"
      rel="noopener noreferrer"
      class="text-accent hover:underline"
    >Open the {{ cluster }} dashboard ↗</a>
    <span v-else>Open the {{ cluster }} dashboard to view its logs.</span>
  </div>
</template>
