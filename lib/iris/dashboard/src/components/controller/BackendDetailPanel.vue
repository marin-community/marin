<script setup lang="ts">
import { computed } from 'vue'
import type { BackendSummary } from '@/types/rpc'
import { RouterLink } from 'vue-router'

// Renders the expanded, backend-authored status for one backend. The variant is
// selected by the backend's capability: a cluster-view backend carries
// `detail.kubernetes`; a worker-daemon backend carries `detail.worker`.
const props = defineProps<{ backend: BackendSummary }>()

const kubernetes = computed(() => props.backend.detail?.kubernetes)
const worker = computed(() => props.backend.detail?.worker)
const scaleGroupCount = computed(() => props.backend.scaleGroups?.length ?? 0)
</script>

<template>
  <div class="border-t border-surface-border bg-surface-raised px-4 py-4">
    <div class="grid sm:grid-cols-3 gap-3 text-sm">
      <div class="p-3 border border-surface-border rounded bg-surface">
        <div class="text-xs text-text-muted">Registered RPC nodes</div>
        <div class="font-semibold">{{ backend.workerCount }}</div>
      </div>
      <div class="p-3 border border-surface-border rounded bg-surface">
        <div class="text-xs text-text-muted">Scale groups</div>
        <div class="font-semibold">{{ scaleGroupCount }}</div>
      </div>
      <div class="p-3 border border-surface-border rounded bg-surface">
        <div class="text-xs text-text-muted">Running tasks</div>
        <div class="font-semibold">{{ backend.runningTaskCount }}</div>
      </div>
    </div>
    <div class="mt-3 flex gap-3 text-sm">
      <RouterLink :to="{ path: '/nodes', query: { backend: backend.backendId } }" class="text-accent hover:underline">View nodes</RouterLink>
      <RouterLink :to="{ path: '/capacity', query: { backend: backend.backendId } }" class="text-accent hover:underline">View slices</RouterLink>
      <span v-if="worker" class="text-text-muted">{{ worker.healthyWorkerCount ?? 0 }}/{{ worker.totalWorkerCount ?? 0 }} healthy</span>
    </div>
    <div v-if="kubernetes" class="mt-4 border-t border-surface-border pt-3 text-sm space-y-2">
      <div class="text-text-secondary">
        Kubernetes namespace <span class="font-mono">{{ kubernetes.namespace || 'default' }}</span>
        · {{ kubernetes.schedulableNodes ?? 0 }}/{{ kubernetes.totalNodes ?? 0 }} schedulable nodes
      </div>
      <div v-if="(kubernetes.nodePools ?? []).length" class="flex flex-wrap gap-2">
        <span v-for="pool in kubernetes.nodePools" :key="pool.name" class="px-2 py-1 rounded border border-surface-border bg-surface font-mono text-xs">
          {{ pool.name }}: {{ pool.currentNodes }}/{{ pool.targetNodes }} nodes
        </span>
      </div>
      <div v-if="(kubernetes.podStatuses ?? []).length" class="text-xs text-text-muted">
        {{ kubernetes.podStatuses?.length }} managed pod{{ kubernetes.podStatuses?.length === 1 ? '' : 's' }} observed
      </div>
    </div>
  </div>
</template>
