<script setup lang="ts">
// A federation peer/cluster name that links inward to the parent's own jobs list
// filtered to that cluster (?cluster=<peer>). It deliberately does not link out
// to the child cluster's own dashboard. Used wherever a federated job or task
// surfaces its owning cluster.
defineProps<{
  cluster: string
}>()
</script>

<template>
  <RouterLink :to="{ path: '/', query: { cluster } }" class="font-mono text-accent hover:underline">{{ cluster }}</RouterLink>
</template>
