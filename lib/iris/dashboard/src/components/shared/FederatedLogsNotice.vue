<script setup lang="ts">
// Shown in place of the log viewer for a federated job or task: its logs are
// written on the peer cluster, not the local log store, so the local proxy has
// nothing to serve. Renders an empty state with no outbound link — users can't
// reach a peer's dashboard.
defineProps<{
  cluster: string
  subject: string
}>()
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface px-4 py-3 text-sm text-text-secondary">
    This {{ subject }} runs on the <span class="font-mono text-text">{{ cluster }}</span> cluster.
    Its logs aren't surfaced in this view yet.
  </div>
</template>
