<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import Shell from '@marina/Shell.vue'
import RefreshButton from '@/components/shared/RefreshButton.vue'
import TooltipHost from '@/components/shared/TooltipHost.vue'
import { useAutoRefresh } from '@/composables/useRefresh'

useAutoRefresh()
</script>

<template>
  <Shell app="evaldash">
    <template #nav>
      <RouterLink to="/">Panel</RouterLink>
      <RouterLink to="/models">Models</RouterLink>
      <RouterLink to="/compare">Compare</RouterLink>
      <RouterLink to="/runs">Runs</RouterLink>
      <RouterLink to="/inspect">Inspect</RouterLink>
      <RouterLink to="/debug">Debug</RouterLink>
      <RefreshButton />
    </template>
    <main class="flex-1 px-6 py-4 max-w-[1600px] w-full mx-auto">
      <RouterView />
    </main>
    <TooltipHost />
  </Shell>
</template>
