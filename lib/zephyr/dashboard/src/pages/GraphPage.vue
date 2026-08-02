<script setup lang="ts">
import { onMounted, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import PlanGraph from '@/components/PlanGraph.vue'
import SummaryCards from '@/components/SummaryCards.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId } from '@/composables/usePipelineSelection'
import { useDashboardRpc } from '@/composables/useRpc'
import type { PipelinePlan, PipelineStatus } from '@/types/dashboard'

const planRpc = useDashboardRpc<PipelinePlan>('GetPlan', () => ({ executionId: selectedExecutionId.value }))
const statusRpc = useDashboardRpc<PipelineStatus>('GetStatus', () => ({ executionId: selectedExecutionId.value }))

async function refresh() {
  await Promise.all([planRpc.refresh(), statusRpc.refresh()])
}

useAutoRefresh(refresh)
onMounted(refresh)
watch(selectedExecutionId, () => void refresh())
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="planRpc.error.value || statusRpc.error.value" />
    <SummaryCards :plan="planRpc.data.value" :status="statusRpc.data.value" />
    <div v-if="statusRpc.data.value?.fatalError" class="rounded-lg border border-status-danger bg-status-danger-bg p-4">
      <p class="text-xs font-semibold uppercase tracking-wide text-status-danger">Pipeline error</p>
      <pre class="mt-2 whitespace-pre-wrap text-xs text-status-danger">{{ statusRpc.data.value.fatalError }}</pre>
    </div>
    <PlanGraph v-if="planRpc.data.value" :plan="planRpc.data.value" :status="statusRpc.data.value" />
    <div v-else class="card grid min-h-96 place-items-center text-sm text-text-secondary">Loading pipeline plan…</div>
  </div>
</template>
