<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import StateBadge from '@/components/StateBadge.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId, selectPipeline } from '@/composables/usePipelineSelection'
import { useDashboardApi } from '@/composables/useApi'
import type { PipelineList, PipelineStatus } from '@/types/dashboard'
import { irisTaskHref } from '@/utils/formatting'

const pipelinesApi = useDashboardApi<PipelineList>('pipelines')
const statusApi = useDashboardApi<PipelineStatus>('status', () => ({
  execution_id: selectedExecutionId.value,
}))

const pipelines = computed(() => pipelinesApi.data.value?.pipelines ?? [])
const selectedPipeline = computed(() =>
  pipelines.value.find((pipeline) => pipeline.execution_id === selectedExecutionId.value),
)
const title = computed(() => selectedPipeline.value?.pipeline_name || 'Zephyr coordinator')
const subtitle = computed(() => {
  const pipeline = selectedPipeline.value
  if (!pipeline) return `${pipelines.value.length} active pipelines`
  return `${pipeline.execution_id} · ${pipelines.value.length} active pipeline${pipelines.value.length === 1 ? '' : 's'}`
})

async function refresh() {
  await pipelinesApi.refresh()
  const available = pipelines.value
  if (!available.some((pipeline) => pipeline.execution_id === selectedExecutionId.value)) {
    selectPipeline(available[0]?.execution_id ?? '')
  }
  await statusApi.refresh()
}

const polling = useAutoRefresh(refresh)
onMounted(refresh)
watch(selectedExecutionId, () => void statusApi.refresh())

function toggleTheme() {
  const dark = document.documentElement.classList.toggle('dark')
  localStorage.setItem('zephyr-dark-mode', String(dark))
}
</script>

<template>
  <header class="border-b border-surface-border bg-surface-raised">
    <div class="mx-auto flex min-h-20 max-w-[1600px] items-center gap-4 px-4 sm:px-6 lg:px-8">
      <div class="grid h-10 w-10 place-items-center rounded-xl bg-accent text-lg font-black text-white">Z</div>
      <div class="min-w-0 flex-1">
        <div class="flex items-center gap-3">
          <h1 class="truncate text-lg font-semibold">{{ title }}</h1>
          <StateBadge :value="statusApi.data.value?.phase" />
        </div>
        <p class="truncate font-mono text-xs text-text-secondary">{{ subtitle }}</p>
      </div>
      <label class="hidden min-w-64 flex-col gap-1 text-[11px] text-text-muted md:flex">
        Active pipeline
        <select
          :value="selectedExecutionId"
          class="rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-sm font-medium text-text"
          @change="selectPipeline(($event.target as HTMLSelectElement).value)"
        >
          <option v-if="!pipelines.length" value="">No active pipelines</option>
          <option v-for="pipeline in pipelines" :key="pipeline.execution_id" :value="pipeline.execution_id">
            {{ pipeline.pipeline_name || pipeline.execution_id }} · {{ pipeline.current_stage || 'starting' }}
          </option>
        </select>
      </label>
      <button
        class="rounded-lg border border-surface-border px-3 py-2 text-xs font-medium text-text-secondary hover:bg-surface-sunken"
        :title="polling.active.value ? 'Pause live updates' : 'Resume live updates'"
        @click="polling.toggle"
      >
        <span :class="['mr-1.5 inline-block h-2 w-2 rounded-full', polling.active.value ? 'bg-status-success' : 'bg-text-muted']" />
        {{ polling.active.value ? 'Live' : 'Paused' }}
      </button>
      <a
        v-if="statusApi.data.value?.coordinator_task_id"
        :href="irisTaskHref(statusApi.data.value.coordinator_task_id)"
        target="_top"
        class="hidden rounded-lg border border-surface-border px-3 py-2 text-xs font-medium text-text-secondary hover:bg-surface-sunken sm:block"
      >
        Coordinator task
      </a>
      <button
        class="grid h-9 w-9 place-items-center rounded-lg border border-surface-border text-text-secondary hover:bg-surface-sunken"
        title="Change theme"
        @click="toggleTheme"
      >
        ◐
      </button>
    </div>
  </header>
</template>
