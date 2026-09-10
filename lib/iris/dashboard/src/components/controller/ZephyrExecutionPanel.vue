<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useLogServerStatsRpc } from '@/composables/useRpc'
import { DEFAULT_REFRESH_MS, useAutoRefresh } from '@/composables/useAutoRefresh'
import { decodeArrowIpc } from '@/utils/arrow'
import { executionPlansSql, EXECUTION_LIMIT, type ExecutionPlan } from '@/utils/zephyr'
import ZephyrExecutionStages from './ZephyrExecutionStages.vue'

const props = defineProps<{ jobId: string; startMs: number }>()
const selectedExecution = ref('')
const namespaces = useLogServerStatsRpc<{ namespaces?: { namespace: string }[] }>('ListNamespaces')
const plans = useLogServerStatsRpc<{ arrowIpc?: string }>('Query', () => ({
  sql: executionPlansSql(props.jobId, props.startMs),
}))
const available = computed(() => (namespaces.data.value?.namespaces ?? []).map(item => item.namespace))
const executions = computed(() => decodeArrowIpc(plans.data.value?.arrowIpc).rows as unknown as ExecutionPlan[])
const execution = computed(() => executions.value.find(item => item.execution_id === selectedExecution.value))
const error = computed(() => namespaces.error.value ?? plans.error.value)

async function refresh() {
  if (!props.startMs || namespaces.loading.value || plans.loading.value) return
  await namespaces.refresh()
  if (!available.value.includes('zephyr.execution')) return
  await plans.refresh()
  if (!execution.value) selectedExecution.value = executions.value[0]?.execution_id ?? ''
}

onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
</script>

<template>
  <section v-if="executions.length || error" class="mb-6 rounded-lg border border-surface-border bg-surface p-4" aria-label="Zephyr executions">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <h3 class="text-sm font-semibold uppercase tracking-wider text-text-secondary">Zephyr executions</h3>
      <button class="text-sm text-accent hover:underline" :disabled="plans.loading.value" @click="refresh">Refresh executions</button>
    </div>
    <p v-if="error" role="alert" class="mt-3 text-sm text-red-500">Zephyr telemetry unavailable: {{ error }}</p>
    <template v-if="executions.length">
      <label class="mt-3 flex flex-wrap items-center gap-3 text-sm text-text-secondary">
        Execution
        <select v-model="selectedExecution" class="rounded border border-surface-border bg-surface-sunken px-2 py-1 font-mono text-text">
          <option v-for="item in executions" :key="item.execution_id" :value="item.execution_id">{{ item.execution_id }}</option>
        </select>
        <span v-if="executions.length === EXECUTION_LIMIT">Newest {{ EXECUTION_LIMIT }} executions</span>
      </label>
      <ZephyrExecutionStages v-if="execution" :key="execution.execution_id" :execution="execution" :start-ms="startMs" :namespaces="available" />
    </template>
  </section>
</template>
