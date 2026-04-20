<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { EndpointInfo, ListEndpointsResponse } from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'

const SHOW_ALL_THRESHOLD = 100

const prefix = ref('')
const localPrefix = ref('')
const showAll = ref(false)

const {
  data: listResponse,
  loading,
  error,
  refresh: fetchEndpoints,
} = useControllerRpc<ListEndpointsResponse>('ListEndpoints', () => ({
  prefix: prefix.value || undefined,
}))

const endpoints = computed(() => listResponse.value?.endpoints ?? [])

watch(listResponse, () => { showAll.value = false })

onMounted(fetchEndpoints)
useAutoRefresh(fetchEndpoints, DEFAULT_REFRESH_MS)

function handleFilterSubmit() {
  prefix.value = localPrefix.value
}

function handleFilterClear() {
  localPrefix.value = ''
  prefix.value = ''
}

const visibleEndpoints = computed(() => {
  if (showAll.value || endpoints.value.length <= SHOW_ALL_THRESHOLD) {
    return endpoints.value
  }
  return endpoints.value.slice(0, SHOW_ALL_THRESHOLD)
})

const hasMore = computed(() => endpoints.value.length > SHOW_ALL_THRESHOLD && !showAll.value)

function metadataString(metadata?: Record<string, string>): string {
  if (!metadata) return '-'
  const entries = Object.entries(metadata)
  if (entries.length === 0) return '-'
  return entries.map(([k, v]) => `${k}=${v}`).join(', ')
}

function jobIdFromTaskId(taskId?: string): string | null {
  if (!taskId) return null
  // taskId format: jobId/taskIndex or jobId
  const slash = taskId.lastIndexOf('/')
  return slash > 0 ? taskId.slice(0, slash) : taskId
}
</script>

<template>
  <!-- Filter bar -->
  <div class="mb-4 flex items-center gap-3">
    <form class="flex gap-2" @submit.prevent="handleFilterSubmit">
      <input
        v-model="localPrefix"
        type="text"
        placeholder="Filter by prefix..."
        class="w-52 px-3 py-1.5 text-sm border border-surface-border rounded
               bg-surface placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <button
        type="submit"
        class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised"
      >
        Filter
      </button>
      <button
        v-if="prefix"
        type="button"
        class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-status-danger"
        @click="handleFilterClear"
      >
        Clear
      </button>
    </form>
    <span class="text-[13px] text-text-secondary">
      {{ endpoints.length }} endpoint{{ endpoints.length !== 1 ? 's' : '' }}
    </span>
  </div>

  <!-- Error -->
  <div
    v-if="error"
    class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Loading -->
  <div v-if="loading && endpoints.length === 0" class="flex items-center justify-center py-12 text-text-muted text-sm">
    <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
    Loading...
  </div>

  <!-- Empty state -->
  <EmptyState
    v-else-if="!loading && endpoints.length === 0"
    icon="⬛"
    :message="prefix ? 'No endpoints matching prefix' : 'No endpoints registered'"
  />

  <!-- Endpoints table -->
  <div v-else class="overflow-x-auto">
    <table class="w-full border-collapse">
      <thead>
        <tr class="border-b border-surface-border">
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Name
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Address
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Job
          </th>
          <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">
            Metadata
          </th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="ep in visibleEndpoints"
          :key="ep.endpointId ?? ep.name"
          class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
        >
          <td class="px-3 py-2 text-[13px] font-mono">{{ ep.name }}</td>
          <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
            <span v-if="ep.address" class="group/addr inline-flex items-center gap-1">
              {{ ep.address }}
              <CopyButton :value="ep.address" />
            </span>
            <span v-else>-</span>
          </td>
          <td class="px-3 py-2 text-[13px]">
            <RouterLink
              v-if="ep.taskId && jobIdFromTaskId(ep.taskId)"
              :to="'/job/' + encodeURIComponent(jobIdFromTaskId(ep.taskId)!)"
              class="font-mono text-accent hover:underline text-xs"
            >
              {{ jobIdFromTaskId(ep.taskId) }}
            </RouterLink>
            <span v-else class="text-text-muted">-</span>
          </td>
          <td class="px-3 py-2 text-xs text-text-muted font-mono max-w-xs truncate" :title="metadataString(ep.metadata)">
            {{ metadataString(ep.metadata) }}
          </td>
        </tr>
      </tbody>
    </table>

    <!-- Show all toggle -->
    <div v-if="hasMore || (endpoints.length > SHOW_ALL_THRESHOLD && showAll)" class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
      <span>Showing {{ visibleEndpoints.length }} of {{ endpoints.length }}</span>
      <button
        class="ml-3 text-accent hover:underline"
        @click="showAll = !showAll"
      >
        {{ showAll ? 'Show first 100' : 'Show all' }}
      </button>
    </div>
  </div>
</template>
