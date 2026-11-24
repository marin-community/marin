<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import Shell from '@marina/Shell.vue'
import TooltipHost from '@/components/shared/TooltipHost.vue'
import { useAutoRefresh } from '@/composables/useRefresh'

const route = useRoute()
const cohortQuery = computed(() => ({ cohort: route.query.cohort }))
const toolsMenu = ref<HTMLDetailsElement | null>(null)

function closeTools() {
  if (toolsMenu.value) toolsMenu.value.open = false
}

watch(() => route.fullPath, closeTools)
useAutoRefresh()
</script>

<template>
  <Shell app="evaldash">
    <template #nav>
      <RouterLink to="/runs" :aria-current="['runs', 'run', 'samples'].includes(String(route.name)) ? 'page' : undefined">Runs</RouterLink>
      <RouterLink :to="{ name: 'compare', query: cohortQuery }">Compare</RouterLink>
      <RouterLink :to="{ name: 'benchmarks', query: cohortQuery }" :aria-current="['benchmarks', 'panel'].includes(String(route.name)) ? 'page' : undefined">Benchmarks</RouterLink>
      <RouterLink :to="{ name: 'models', query: cohortQuery }" :aria-current="['models', 'model'].includes(String(route.name)) ? 'page' : undefined">Models</RouterLink>
      <details ref="toolsMenu" class="tools-menu" @keydown.esc="closeTools(); toolsMenu?.querySelector('summary')?.focus()" @focusout="!toolsMenu?.contains($event.relatedTarget as Node) && closeTools()">
        <summary>More</summary>
        <div>
          <RouterLink :to="{ name: 'reports' }">Playground reports</RouterLink>
          <RouterLink :to="{ name: 'debug' }">Diagnostics</RouterLink>
        </div>
      </details>
    </template>
    <div class="review-page" @click="closeTools">
      <RouterView />
    </div>
    <TooltipHost />
  </Shell>
</template>
