<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import Shell from '@marina/Shell.vue'
import TooltipHost from '@/components/shared/TooltipHost.vue'
import { useAutoRefresh } from '@/composables/useRefresh'

const route = useRoute()
const cohortQuery = computed(() => ({ cohort: route.query.cohort }))
useAutoRefresh()
</script>

<template>
  <Shell app="evaldash">
    <template #nav>
      <RouterLink to="/runs" :aria-current="['runs', 'run', 'samples'].includes(String(route.name)) ? 'page' : undefined">Runs</RouterLink>
      <RouterLink :to="{ name: 'compare', query: cohortQuery }">Compare</RouterLink>
      <RouterLink :to="{ name: 'benchmarks', query: cohortQuery }" :aria-current="['benchmarks', 'panel'].includes(String(route.name)) ? 'page' : undefined">Benchmarks</RouterLink>
      <RouterLink :to="{ name: 'models', query: cohortQuery }" :aria-current="['models', 'model'].includes(String(route.name)) ? 'page' : undefined">Models</RouterLink>
      <RouterLink :to="{ name: 'reports' }" :aria-current="['reports', 'report'].includes(String(route.name)) ? 'page' : undefined">Reports</RouterLink>
      <RouterLink :to="{ name: 'debug' }" class="diagnostics-link" aria-label="Diagnostics" title="Diagnostics">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true">
          <path d="M2 12h5l3-8 4 16 3-8h5" />
        </svg>
        <span>Diagnostics</span>
      </RouterLink>
    </template>
    <div class="review-page">
      <RouterView />
    </div>
    <TooltipHost />
  </Shell>
</template>
