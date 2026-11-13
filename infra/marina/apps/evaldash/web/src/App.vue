<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import Shell from '@marina/Shell.vue'
import RefreshButton from '@/components/shared/RefreshButton.vue'
import TooltipHost from '@/components/shared/TooltipHost.vue'
import { useAutoRefresh } from '@/composables/useRefresh'

const route = useRoute()
const cohortQuery = computed(() => ({ cohort: route.query.cohort }))

useAutoRefresh()
</script>

<template>
  <Shell app="evaldash">
    <template #nav>
      <RouterLink :to="{ path: '/', query: cohortQuery }">Panel</RouterLink>
      <RouterLink :to="{ path: '/models', query: cohortQuery }">Models</RouterLink>
      <RouterLink :to="{ path: '/compare', query: cohortQuery }">Compare</RouterLink>
      <RouterLink :to="{ path: '/runs', query: cohortQuery }">Runs</RouterLink>
      <RouterLink :to="{ path: '/inspect', query: cohortQuery }">Inspect</RouterLink>
      <RouterLink :to="{ path: '/debug', query: cohortQuery }">Debug</RouterLink>
      <RefreshButton />
    </template>
    <main class="flex-1 px-6 py-4 max-w-[1600px] w-full mx-auto">
      <RouterView />
    </main>
    <TooltipHost />
  </Shell>
</template>
