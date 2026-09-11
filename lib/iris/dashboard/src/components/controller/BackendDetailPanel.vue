<script setup lang="ts">
import { computed } from 'vue'
import type { BackendSummary } from '@/types/rpc'
import FleetOverview from '@/components/controller/FleetOverview.vue'
import KubernetesClusterDetail from '@/components/controller/KubernetesClusterDetail.vue'
import EmptyState from '@/components/shared/EmptyState.vue'

// Renders the expanded, backend-authored status for one backend. The variant is
// selected by the backend's capability: a cluster-view backend carries
// `detail.kubernetes`; a worker-daemon backend carries `detail.worker`.
const props = defineProps<{ backend: BackendSummary }>()

const kubernetes = computed(() => props.backend.detail?.kubernetes)
const worker = computed(() => props.backend.detail?.worker)
const fleetGroups = computed(() => worker.value?.autoscaler?.groups ?? [])
</script>

<template>
  <div class="border-t border-surface-border bg-surface-raised px-4 py-4">
    <KubernetesClusterDetail v-if="kubernetes" :status="kubernetes" />

    <div v-else-if="worker" class="space-y-4">
      <div class="flex items-center gap-2 text-sm">
        <span
          class="w-2 h-2 rounded-full shrink-0"
          :class="(worker.healthyWorkerCount ?? 0) === (worker.totalWorkerCount ?? 0)
            ? 'bg-status-success' : 'bg-status-warning'"
        />
        <span class="text-text-secondary">Workers healthy:</span>
        <span class="font-semibold font-mono tabular-nums">
          {{ worker.healthyWorkerCount ?? 0 }} / {{ worker.totalWorkerCount ?? 0 }}
        </span>
      </div>
      <FleetOverview
        v-if="fleetGroups.length > 0"
        :groups="fleetGroups"
        :running-buckets="[]"
      />
      <EmptyState v-else message="No scale groups for this backend." />
    </div>

    <EmptyState v-else message="No detail reported for this backend." />
  </div>
</template>
