<script setup lang="ts">
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'

// Phase 1 shell: sidebar + run selection. RunHeader / Tabs / charts / profile
// land in later phases (see .agents/projects/2026-07-01_buoy_vue_dashboard_migration.md).
const selectedRun = ref<string | null>(null)
</script>

<template>
  <div class="flex h-screen">
    <Sidebar @select="(name) => (selectedRun = name)" />
    <main class="flex-1 overflow-auto p-6">
      <p v-if="!selectedRun" class="text-text-muted">pick an entity / project, then a run from the list</p>
      <p v-else class="break-all font-mono text-lg font-semibold text-accent">{{ selectedRun }}</p>
    </main>
  </div>
</template>
