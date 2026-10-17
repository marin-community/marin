<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useEndpointRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { EndpointInfo, ListEndpointsResponse } from '@/types/rpc'
import EmptyState from '@/components/shared/EmptyState.vue'
import CopyButton from '@/components/shared/CopyButton.vue'
import EndpointLink from '@/components/shared/EndpointLink.vue'
import { filterEndpoints, groupEndpointsByOwner, sortEndpointsByName } from '@/utils/endpoints'

const SHOW_ALL_THRESHOLD = 100

const query = ref('')
const showAll = ref(false)

const {
  data: taskListResponse,
  loading: taskLoading,
  error: taskError,
  refresh: fetchTaskEndpoints,
} = useEndpointRpc<ListEndpointsResponse>('ListEndpoints')
const {
  data: systemListResponse,
  loading: systemLoading,
  error: systemError,
  refresh: fetchSystemEndpoints,
} = useEndpointRpc<ListEndpointsResponse>('ListEndpoints', { prefix: '/system/' })

const endpoints = computed(() => [
  ...(systemListResponse.value?.endpoints ?? []),
  ...(taskListResponse.value?.endpoints ?? []),
])
const loading = computed(() => taskLoading.value || systemLoading.value)
const error = computed(() => taskError.value || systemError.value)
const matchingEndpoints = computed(() => sortEndpointsByName(filterEndpoints(endpoints.value, query.value)))

watch(() => matchingEndpoints.value.length, () => { showAll.value = false })

async function fetchEndpoints() {
  await Promise.all([fetchTaskEndpoints(), fetchSystemEndpoints()])
}

onMounted(fetchEndpoints)
useAutoRefresh(fetchEndpoints, DEFAULT_REFRESH_MS)

const visibleEndpoints = computed(() => {
  if (showAll.value || matchingEndpoints.value.length <= SHOW_ALL_THRESHOLD) {
    return matchingEndpoints.value
  }
  return matchingEndpoints.value.slice(0, SHOW_ALL_THRESHOLD)
})

const groupedEndpoints = computed(() => groupEndpointsByOwner(visibleEndpoints.value))
const hasMore = computed(() => matchingEndpoints.value.length > SHOW_ALL_THRESHOLD && !showAll.value)

function metadataString(metadata?: Record<string, string>): string {
  if (!metadata) return '-'
  const entries = Object.entries(metadata)
  if (entries.length === 0) return '-'
  return entries.map(([k, v]) => `${k}=${v}`).join(', ')
}

</script>

<template>
  <!-- Filter bar -->
  <div class="mb-4 flex items-center gap-3">
    <div class="flex gap-2">
      <input
        v-model="query"
        type="text"
        placeholder="Search endpoints..."
        aria-label="Search endpoints"
        class="w-64 px-3 py-1.5 text-sm border border-surface-border rounded
               bg-surface placeholder:text-text-muted
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <button
        v-if="query"
        type="button"
        class="px-3 py-1.5 text-sm border border-surface-border rounded hover:bg-surface-raised text-status-danger"
        @click="query = ''"
      >
        Clear
      </button>
    </div>
    <span class="text-[13px] text-text-secondary">
      <template v-if="query.trim()">
        {{ matchingEndpoints.length }} of
      </template>
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
    v-else-if="!loading && matchingEndpoints.length === 0"
    icon="⬛"
    :message="query.trim() ? 'No endpoints match this search' : 'No endpoints registered'"
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
            Metadata
          </th>
        </tr>
      </thead>
      <tbody>
        <template v-for="userGroup in groupedEndpoints" :key="userGroup.user ?? 'system'">
          <tr class="border-b border-surface-border bg-surface-sunken">
            <th colspan="3" class="px-3 py-2 text-left text-xs font-semibold text-text">
              <RouterLink
                v-if="userGroup.user"
                :to="{ path: '/', query: { user: userGroup.user } }"
                class="text-accent hover:underline"
              >
                {{ userGroup.user }}
              </RouterLink>
              <span v-else>System</span>
              <span class="ml-1 font-normal text-text-muted">
                {{ userGroup.endpointCount }} endpoint{{ userGroup.endpointCount !== 1 ? 's' : '' }}
              </span>
            </th>
          </tr>
          <template v-for="jobGroup in userGroup.jobs" :key="jobGroup.jobId ?? 'system'">
            <tr v-if="jobGroup.jobId" class="border-b border-surface-border-subtle bg-surface-raised/40">
              <th colspan="3" class="py-1.5 pl-6 pr-3 text-left text-xs font-normal">
                <RouterLink
                  :to="'/job/' + encodeURIComponent(jobGroup.jobId)"
                  class="font-mono text-accent hover:underline"
                >
                  {{ jobGroup.jobId }}
                </RouterLink>
              </th>
            </tr>
            <tr
              v-for="ep in jobGroup.endpoints"
              :key="ep.endpointId ?? ep.name"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <td :class="['py-2 pr-3 text-[13px] font-mono', jobGroup.jobId ? 'pl-9' : 'pl-3']">
                <EndpointLink :name="ep.name" />
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">
                <span v-if="ep.address" class="group/addr inline-flex items-center gap-1">
                  {{ ep.address }}
                  <CopyButton :value="ep.address" />
                </span>
                <span v-else>-</span>
              </td>
              <td class="px-3 py-2 text-xs text-text-muted font-mono max-w-xs truncate" :title="metadataString(ep.metadata)">
                {{ metadataString(ep.metadata) }}
              </td>
            </tr>
          </template>
        </template>
      </tbody>
    </table>

    <!-- Show all toggle -->
    <div v-if="hasMore || (matchingEndpoints.length > SHOW_ALL_THRESHOLD && showAll)" class="px-3 py-2 text-xs text-text-secondary border-t border-surface-border">
      <span>Showing {{ visibleEndpoints.length }} of {{ matchingEndpoints.length }}</span>
      <button
        class="ml-3 text-accent hover:underline"
        @click="showAll = !showAll"
      >
        {{ showAll ? 'Show first 100' : 'Show all' }}
      </button>
    </div>
  </div>
</template>
