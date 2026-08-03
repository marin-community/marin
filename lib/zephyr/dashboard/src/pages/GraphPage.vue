<script setup lang="ts">
import { onMounted, watch } from 'vue'
import ErrorBanner from '@/components/ErrorBanner.vue'
import PlanGraph from '@/components/PlanGraph.vue'
import SummaryCards from '@/components/SummaryCards.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { selectedExecutionId } from '@/composables/usePipelineSelection'
import { useDashboardApi } from '@/composables/useApi'
import type { PipelinePlan, PipelineStatus } from '@/types/dashboard'

const planApi = useDashboardApi<PipelinePlan>('plan', () => ({ execution_id: selectedExecutionId.value }))
const statusApi = useDashboardApi<PipelineStatus>('status', () => ({ execution_id: selectedExecutionId.value }))

async function refresh() {
  await Promise.all([planApi.refresh(), statusApi.refresh()])
}

useAutoRefresh(refresh)
onMounted(refresh)
watch(selectedExecutionId, () => void refresh())
</script>

<template>
  <div class="space-y-4">
    <ErrorBanner :message="planApi.error.value || statusApi.error.value" />
    <SummaryCards :plan="planApi.data.value" :status="statusApi.data.value" />
    <div v-if="statusApi.data.value?.fatal_error" class="rounded-lg border border-status-danger bg-status-danger-bg p-4">
      <p class="text-xs font-semibold uppercase tracking-wide text-status-danger">Pipeline error</p>
      <pre class="mt-2 whitespace-pre-wrap text-xs text-status-danger">{{ statusApi.data.value.fatal_error }}</pre>
    </div>
    <PlanGraph v-if="planApi.data.value" :plan="planApi.data.value" :status="statusApi.data.value" />
    <div v-else class="card grid min-h-96 place-items-center text-sm text-text-secondary">Loading pipeline plan…</div>
  </div>
</template>
