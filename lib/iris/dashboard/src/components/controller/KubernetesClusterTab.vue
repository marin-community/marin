<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useControllerRpc } from '@/composables/useRpc'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import { timestampMs, formatRelativeTime } from '@/utils/formatting'
import MetricCard from '@/components/shared/MetricCard.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import type { GetKubernetesClusterStatusResponse, NodePoolStatus } from '@/types/rpc'

const { data, loading, error, refresh } = useControllerRpc<GetKubernetesClusterStatusResponse>('GetKubernetesClusterStatus')
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)
onMounted(refresh)

// -- Pod phase styling --

function phaseClass(phase: string): string {
  switch (phase) {
    case 'Running': return 'text-status-success'
    case 'Succeeded': return 'text-accent'
    case 'Failed': return 'text-status-danger'
    case 'Pending': return 'text-status-warning'
    default: return 'text-text-muted'
  }
}

function phaseDotClass(phase: string): string {
  switch (phase) {
    case 'Running': return 'bg-status-success'
    case 'Succeeded': return 'bg-accent'
    case 'Failed': return 'bg-status-danger'
    case 'Pending': return 'bg-status-warning animate-pulse'
    default: return 'bg-text-muted'
  }
}

function formatTransition(ts?: { epochMs?: string }): string {
  if (!ts?.epochMs) return '-'
  const ms = timestampMs(ts as { epochMs: string })
  if (!ms) return '-'
  return formatRelativeTime(ms)
}

// -- NodePool helpers --

function poolProvisioningStatus(pool: NodePoolStatus): 'ready' | 'provisioning' | 'scaling' | 'error' {
  if (pool.capacity === 'Insufficient') return 'error'
  if (pool.inProgressNodes > 0 || pool.queuedNodes > 0) return 'provisioning'
  if (pool.currentNodes < pool.targetNodes) return 'scaling'
  return 'ready'
}

function poolStatusLabel(pool: NodePoolStatus): string {
  const s = poolProvisioningStatus(pool)
  switch (s) {
    case 'ready': return 'Ready'
    case 'provisioning': return `Provisioning ${pool.inProgressNodes + pool.queuedNodes} node${pool.inProgressNodes + pool.queuedNodes > 1 ? 's' : ''}`
    case 'scaling': return 'Scaling'
    case 'error': return 'Insufficient capacity'
  }
}

function poolStatusClasses(pool: NodePoolStatus): string {
  const s = poolProvisioningStatus(pool)
  switch (s) {
    case 'ready': return 'bg-status-success-bg text-status-success border-status-success-border'
    case 'provisioning': return 'bg-status-purple-bg text-status-purple border-status-purple-border'
    case 'scaling': return 'bg-status-warning-bg text-status-warning border-status-warning-border'
    case 'error': return 'bg-status-danger-bg text-status-danger border-status-danger-border'
  }
}

function poolProgressPercent(pool: NodePoolStatus): number {
  if (pool.targetNodes === 0) return 100
  return Math.min(100, Math.round((pool.currentNodes / pool.targetNodes) * 100))
}

function poolProgressBarClass(pool: NodePoolStatus): string {
  const s = poolProvisioningStatus(pool)
  switch (s) {
    case 'ready': return 'bg-status-success'
    case 'provisioning': return 'bg-status-purple'
    case 'scaling': return 'bg-status-warning'
    case 'error': return 'bg-status-danger'
  }
}

// -- Computed --

const pools = computed(() => data.value?.nodePools ?? [])
const pods = computed(() => data.value?.podStatuses ?? [])

const provisioningPools = computed(() =>
  pools.value.filter(p => poolProvisioningStatus(p) === 'provisioning')
)

const pendingPods = computed(() =>
  pods.value.filter(p => p.phase === 'Pending')
)

const hasAutoscalerBanner = computed(() =>
  pendingPods.value.length > 0 && provisioningPools.value.length > 0
)

const podsByPhase = computed(() => {
  const counts: Record<string, number> = {}
  for (const pod of pods.value) {
    counts[pod.phase] = (counts[pod.phase] ?? 0) + 1
  }
  return counts
})

function podPhaseSummary(): string {
  const total = pods.value.length
  if (total === 0) return '0'
  const order = ['Running', 'Pending', 'Succeeded', 'Failed']
  const parts: string[] = []
  for (const phase of order) {
    const n = podsByPhase.value[phase] ?? 0
    if (n > 0) parts.push(`${n} ${phase.toLowerCase()}`)
  }
  return `${total} (${parts.join(', ')})`
}

function nodeDisplayName(nodeName?: string, phase?: string): string {
  if (nodeName) return nodeName
  if (phase === 'Pending') return 'Waiting for node'
  return '-'
}

function nodeDisplayClass(nodeName?: string, phase?: string): string {
  if (nodeName) return 'text-text-muted'
  if (phase === 'Pending') return 'text-status-warning italic'
  return 'text-text-muted'
}
</script>

<template>
  <!-- Loading -->
  <div v-if="loading && !data" class="flex items-center justify-center py-12 text-text-muted text-sm">
    Loading cluster status...
  </div>

  <!-- Error -->
  <div
    v-else-if="error"
    class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
  >
    {{ error }}
  </div>

  <!-- Main -->
  <div v-else-if="data" class="space-y-6">

    <!-- ===== Status Bar ===== -->
    <div class="flex flex-wrap items-center gap-3 text-sm">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span
          :class="[
            'w-2 h-2 rounded-full',
            (data.schedulableNodes ?? 0) > 0 ? 'bg-status-success' : 'bg-text-muted',
          ]"
        />
        <span class="text-text-secondary">Nodes:</span>
        <span class="font-semibold font-mono">{{ data.schedulableNodes ?? 0 }} / {{ data.totalNodes ?? 0 }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">CPU:</span>
        <span class="font-semibold font-mono ml-1">{{ data.allocatableCpu || '0' }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Memory:</span>
        <span class="font-semibold font-mono ml-1">{{ data.allocatableMemory || '0' }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Pods:</span>
        <span class="font-semibold font-mono ml-1">{{ podPhaseSummary() }}</span>
      </div>

      <div v-if="pools.length > 0" class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Node Pools:</span>
        <span class="font-semibold font-mono ml-1">{{ pools.length }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Namespace:</span>
        <span class="font-semibold font-mono ml-1">{{ data.namespace || '-' }}</span>
      </div>
    </div>

    <!-- ===== Autoscaler Banner ===== -->
    <div
      v-if="hasAutoscalerBanner"
      class="flex items-center gap-3 px-4 py-3 rounded-lg bg-status-purple-bg border border-status-purple-border"
    >
      <span class="w-2.5 h-2.5 rounded-full bg-status-purple animate-pulse flex-shrink-0" />
      <div class="text-sm">
        <span class="font-semibold text-status-purple">Waiting for autoscaler</span>
        <span class="text-text-secondary ml-1">
          — {{ pendingPods.length }} pod{{ pendingPods.length > 1 ? 's' : '' }} pending,
          {{ provisioningPools.map(p => `${p.inProgressNodes + p.queuedNodes} node${p.inProgressNodes + p.queuedNodes > 1 ? 's' : ''} provisioning on ${p.name}`).join('; ') }}
        </span>
      </div>
    </div>

    <!-- ===== Node Pools ===== -->
    <section v-if="pools.length > 0">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Node Pools
      </h3>

      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Pool</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Instance Type</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Scale Group</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-28">Nodes</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-36">Progress</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Status</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Capacity</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="pool in pools"
              :key="pool.name"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <!-- Pool name -->
              <td class="px-3 py-2 text-[13px] font-semibold">{{ pool.name }}</td>

              <!-- Instance type -->
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">{{ pool.instanceType }}</td>

              <!-- Scale group -->
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ pool.scaleGroup || '-' }}</td>

              <!-- Nodes: current / target (min-max) -->
              <td class="px-3 py-2 text-[13px] text-right font-mono">
                <span :class="pool.currentNodes < pool.targetNodes ? 'text-status-warning' : 'text-text'">
                  {{ pool.currentNodes }}
                </span>
                <span class="text-text-muted"> / {{ pool.targetNodes }}</span>
                <span class="text-text-muted text-xs ml-1">({{ pool.minNodes }}-{{ pool.maxNodes }})</span>
              </td>

              <!-- Progress bar -->
              <td class="px-3 py-2">
                <div class="flex items-center gap-2">
                  <div class="flex-1 h-1.5 rounded-full bg-surface-border overflow-hidden">
                    <div
                      :class="['h-full rounded-full transition-all duration-500', poolProgressBarClass(pool)]"
                      :style="{ width: poolProgressPercent(pool) + '%' }"
                    />
                  </div>
                  <span class="text-xs font-mono text-text-muted w-8 text-right">{{ poolProgressPercent(pool) }}%</span>
                </div>
              </td>

              <!-- Status badge -->
              <td class="px-3 py-2">
                <span
                  :class="[
                    'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold border',
                    poolStatusClasses(pool),
                  ]"
                >
                  <span
                    v-if="poolProvisioningStatus(pool) === 'provisioning'"
                    class="w-1.5 h-1.5 rounded-full bg-status-purple animate-pulse"
                  />
                  {{ poolStatusLabel(pool) }}
                </span>
              </td>

              <!-- Capacity / Quota -->
              <td class="px-3 py-2 text-[13px]">
                <span
                  v-if="pool.capacity"
                  :class="pool.capacity === 'Sufficient' ? 'text-status-success' : 'text-status-danger'"
                >
                  {{ pool.capacity }}
                </span>
                <span v-if="pool.quota" class="text-text-muted ml-1">
                  / {{ pool.quota }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ===== Pod Statuses ===== -->
    <section>
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Pod Statuses
      </h3>

      <EmptyState v-if="pods.length === 0" message="No iris-managed pods found." />

      <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Pod</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Task ID</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-24">Phase</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Node</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Reason</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Message</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-24">Age</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="pod in pods"
              :key="pod.podName"
              :class="[
                'border-b border-surface-border-subtle hover:bg-surface-raised transition-colors',
                pod.phase === 'Failed' ? 'bg-status-danger-bg/30' : '',
              ]"
            >
              <!-- Pod name -->
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary truncate max-w-[200px]" :title="pod.podName">
                {{ pod.podName }}
              </td>

              <!-- Task ID -->
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary truncate max-w-[200px]" :title="pod.taskId">
                {{ pod.taskId || '-' }}
              </td>

              <!-- Phase with dot -->
              <td class="px-3 py-2 text-[13px]">
                <span class="inline-flex items-center gap-1.5">
                  <span :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', phaseDotClass(pod.phase)]" />
                  <span :class="['font-semibold', phaseClass(pod.phase)]">{{ pod.phase }}</span>
                </span>
              </td>

              <!-- Node -->
              <td class="px-3 py-2 text-[13px] font-mono truncate max-w-[180px]" :title="pod.nodeName || undefined">
                <span :class="nodeDisplayClass(pod.nodeName, pod.phase)">
                  {{ nodeDisplayName(pod.nodeName, pod.phase) }}
                </span>
              </td>

              <!-- Reason -->
              <td class="px-3 py-2 text-[13px] text-text-secondary">
                {{ pod.reason || '-' }}
              </td>

              <!-- Message -->
              <td class="px-3 py-2 text-[13px] text-text-secondary truncate max-w-[250px]" :title="pod.message || undefined">
                {{ pod.message || '-' }}
              </td>

              <!-- Age -->
              <td class="px-3 py-2 text-[13px] text-text-muted font-mono">
                {{ formatTransition(pod.lastTransition) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

  </div>
</template>
