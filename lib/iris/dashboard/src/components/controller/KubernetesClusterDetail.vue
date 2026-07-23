<script setup lang="ts">
import { computed } from 'vue'
import { timestampMs, formatRelativeTime, formatBytes } from '@/utils/formatting'
import EmptyState from '@/components/shared/EmptyState.vue'
import type { GetKubernetesClusterStatusResponse, NodePoolStatus, NodeStatus } from '@/types/rpc'

// Presentational cluster-status view rendered inside a backend's detail panel on
// the Backends tab. The parent supplies the BackendStatus.kubernetes snapshot;
// this component owns no data fetching.
const props = defineProps<{ status: GetKubernetesClusterStatusResponse }>()

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

// -- Node helpers --
// int64 proto fields arrive as JSON strings; coerce before arithmetic/formatting.

function num(v?: string): number {
  return v ? Number(v) : 0
}

function nodeHasMetrics(node: NodeStatus): boolean {
  return num(node.metricsTs) > 0
}

function nodeHealthLabel(node: NodeStatus): string {
  if (!node.ready) return node.statusSummary || 'NotReady'
  if (!node.schedulable) return 'Cordoned'
  return 'Ready'
}

function nodeHealthClasses(node: NodeStatus): string {
  if (!node.ready) return 'bg-status-danger-bg text-status-danger border-status-danger-border'
  if (!node.schedulable) return 'bg-status-warning-bg text-status-warning border-status-warning-border'
  return 'bg-status-success-bg text-status-success border-status-success-border'
}

function nodeHealthDotClass(node: NodeStatus): string {
  if (!node.ready) return 'bg-status-danger'
  if (!node.schedulable) return 'bg-status-warning'
  return 'bg-status-success'
}

function shortGpuModel(model?: string): string {
  if (!model) return ''
  // "NVIDIA H100 80GB HBM3" -> "H100"; a bare "H100" label passes through.
  return model.replace(/^NVIDIA\s+/i, '').split(/\s+/)[0]
}

function acceleratorLabel(node: NodeStatus): string {
  const count = node.gpuCount ?? 0
  if (count <= 0) return 'CPU'
  const model = shortGpuModel(node.gpuModel)
  return model ? `${count}× ${model}` : `${count}× GPU`
}

function isGpuNode(node: NodeStatus): boolean {
  return (node.gpuCount ?? 0) > 0
}

function usagePercent(used: number, total: number): number {
  if (total <= 0) return 0
  return Math.min(100, Math.round((used / total) * 100))
}

// CPU/memory fill is pressure: green with headroom, warning tight, danger near full.
function pressureBarClass(percent: number): string {
  if (percent >= 90) return 'bg-status-danger'
  if (percent >= 75) return 'bg-status-warning'
  return 'bg-status-success'
}

// GPU temperature thresholds (H100 throttles ~87°C).
function tempClass(celsius: number): string {
  if (celsius >= 85) return 'text-status-danger'
  if (celsius >= 70) return 'text-status-warning'
  return 'text-status-success'
}

function memPercent(node: NodeStatus): number {
  return usagePercent(num(node.memUsedBytes), num(node.memTotalBytes))
}

// -- Computed --

const nodes = computed(() => props.status.nodes ?? [])
const pools = computed(() => props.status.nodePools ?? [])
const pods = computed(() => props.status.podStatuses ?? [])

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
  <div class="space-y-4">
    <!-- ===== Status Bar ===== -->
    <div class="flex flex-wrap items-center gap-3 text-sm">
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span
          :class="[
            'w-2 h-2 rounded-full',
            (status.schedulableNodes ?? 0) > 0 ? 'bg-status-success' : 'bg-text-muted',
          ]"
        />
        <span class="text-text-secondary">Nodes:</span>
        <span class="font-semibold font-mono">{{ status.schedulableNodes ?? 0 }} / {{ status.totalNodes ?? 0 }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">CPU:</span>
        <span class="font-semibold font-mono ml-1">{{ status.allocatableCpu || '0' }}</span>
      </div>

      <div class="px-3 py-1.5 rounded-lg bg-surface border border-surface-border">
        <span class="text-text-secondary">Memory:</span>
        <span class="font-semibold font-mono ml-1">{{ status.allocatableMemory || '0' }}</span>
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
        <span class="font-semibold font-mono ml-1">{{ status.namespace || '-' }}</span>
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

    <!-- ===== Nodes ===== -->
    <section v-if="nodes.length > 0">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">
        Nodes
      </h3>

      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Node</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Health</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left">Accelerator</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-52">GPU</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-40">CPU</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-44">Memory</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-right w-16">Pods</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="node in nodes"
              :key="node.name"
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
            >
              <!-- Node identity -->
              <td class="px-3 py-2">
                <div class="text-[13px] font-mono font-semibold">{{ node.name }}</div>
                <div v-if="node.region || node.instanceType" class="text-xs text-text-muted truncate" :title="node.instanceType">
                  {{ [node.region, node.instanceType].filter(Boolean).join(' · ') }}
                </div>
              </td>

              <!-- Health -->
              <td class="px-3 py-2">
                <span
                  :class="['inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold border', nodeHealthClasses(node)]"
                  :title="node.statusSummary"
                >
                  <span :class="['w-1.5 h-1.5 rounded-full', nodeHealthDotClass(node)]" />
                  {{ nodeHealthLabel(node) }}
                </span>
              </td>

              <!-- Accelerator -->
              <td class="px-3 py-2 text-[13px] font-mono" :class="isGpuNode(node) ? 'text-text' : 'text-text-muted'">
                {{ acceleratorLabel(node) }}
              </td>

              <!-- GPU: utilization bar + HBM used/total + hottest-GPU temp -->
              <td class="px-3 py-2">
                <template v-if="isGpuNode(node) && nodeHasMetrics(node)">
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-1.5 rounded-full bg-surface-border overflow-hidden">
                      <div class="h-full rounded-full bg-accent transition-all duration-500" :style="{ width: Math.round(node.gpuUtilPct ?? 0) + '%' }" />
                    </div>
                    <span class="text-xs font-mono text-text-secondary w-9 text-right">{{ Math.round(node.gpuUtilPct ?? 0) }}%</span>
                  </div>
                  <div class="mt-1 text-xs font-mono text-text-muted">
                    {{ formatBytes(num(node.hbmUsedBytes)) }} / {{ formatBytes(num(node.hbmTotalBytes)) }}
                    <span class="mx-1 text-surface-border">|</span>
                    <span :class="tempClass(node.gpuTempC ?? 0)">{{ Math.round(node.gpuTempC ?? 0) }}°C</span>
                  </div>
                </template>
                <span v-else class="text-xs text-text-muted">—</span>
              </td>

              <!-- CPU: live utilization bar over allocatable core count -->
              <td class="px-3 py-2">
                <div v-if="nodeHasMetrics(node)" class="flex items-center gap-2">
                  <div class="flex-1 h-1.5 rounded-full bg-surface-border overflow-hidden">
                    <div :class="['h-full rounded-full transition-all duration-500', pressureBarClass(node.cpuPct ?? 0)]" :style="{ width: Math.round(node.cpuPct ?? 0) + '%' }" />
                  </div>
                  <span class="text-xs font-mono text-text-secondary w-9 text-right">{{ Math.round(node.cpuPct ?? 0) }}%</span>
                </div>
                <div class="text-xs font-mono text-text-muted" :class="{ 'mt-1': nodeHasMetrics(node) }">
                  {{ Math.round(num(node.cpuMillicores) / 1000) }} cores
                </div>
              </td>

              <!-- Memory: live used/total bar -->
              <td class="px-3 py-2">
                <template v-if="nodeHasMetrics(node) && num(node.memTotalBytes) > 0">
                  <div class="flex items-center gap-2">
                    <div class="flex-1 h-1.5 rounded-full bg-surface-border overflow-hidden">
                      <div :class="['h-full rounded-full transition-all duration-500', pressureBarClass(memPercent(node))]" :style="{ width: memPercent(node) + '%' }" />
                    </div>
                    <span class="text-xs font-mono text-text-secondary w-9 text-right">{{ memPercent(node) }}%</span>
                  </div>
                  <div class="mt-1 text-xs font-mono text-text-muted">
                    {{ formatBytes(num(node.memUsedBytes)) }} / {{ formatBytes(num(node.memTotalBytes)) }}
                  </div>
                </template>
                <span v-else class="text-xs font-mono text-text-muted">{{ formatBytes(num(node.memoryBytes)) }}</span>
              </td>

              <!-- Pods -->
              <td class="px-3 py-2 text-[13px] font-mono text-right">
                <span :class="(node.runningPods ?? 0) > 0 ? 'text-text' : 'text-text-muted'">{{ node.runningPods ?? 0 }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

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
              <td class="px-3 py-2 text-[13px] font-semibold">{{ pool.name }}</td>
              <td class="px-3 py-2 text-[13px] font-mono text-text-secondary">{{ pool.instanceType }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ pool.scaleGroup || '-' }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono">
                <span :class="pool.currentNodes < pool.targetNodes ? 'text-status-warning' : 'text-text'">
                  {{ pool.currentNodes }}
                </span>
                <span class="text-text-muted"> / {{ pool.targetNodes }}</span>
                <span class="text-text-muted text-xs ml-1">({{ pool.minNodes }}-{{ pool.maxNodes }})</span>
              </td>
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
        <table class="w-full border-collapse lg:table-fixed">
          <thead>
            <tr class="border-b border-surface-border bg-surface">
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left lg:w-[22%]">Pod</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left lg:w-[20%]">Task ID</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-24 lg:w-[8%]">Phase</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left lg:w-[16%]">Node</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left lg:w-[24%]">Reason</th>
              <th class="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-text-secondary text-left w-24">Age</th>
            </tr>
          </thead>
          <tbody>
            <template
              v-for="pod in pods"
              :key="pod.podName"
            >
              <tr
                :class="[
                  pod.message ? '' : 'border-b border-surface-border-subtle',
                  'hover:bg-surface-raised transition-colors',
                  pod.phase === 'Failed' ? 'bg-status-danger-bg/30' : '',
                ]"
              >
                <td class="px-3 pt-2 pb-1 text-[13px] font-mono text-text-secondary truncate" :title="pod.podName">
                  {{ pod.podName }}
                </td>
                <td class="px-3 pt-2 pb-1 text-[13px] font-mono text-text-secondary truncate" :title="pod.taskId">
                  {{ pod.taskId || '-' }}
                </td>
                <td class="px-3 pt-2 pb-1 text-[13px]">
                  <span class="inline-flex items-center gap-1.5">
                    <span :class="['w-1.5 h-1.5 rounded-full flex-shrink-0', phaseDotClass(pod.phase)]" />
                    <span :class="['font-semibold', phaseClass(pod.phase)]">{{ pod.phase }}</span>
                  </span>
                </td>
                <td class="px-3 pt-2 pb-1 text-[13px] font-mono truncate" :title="pod.nodeName || undefined">
                  <span :class="nodeDisplayClass(pod.nodeName, pod.phase)">
                    {{ nodeDisplayName(pod.nodeName, pod.phase) }}
                  </span>
                </td>
                <td class="px-3 pt-2 pb-1 text-[13px] text-text-secondary truncate" :title="pod.reason || undefined">
                  {{ pod.reason || '-' }}
                </td>
                <td class="px-3 pt-2 pb-1 text-[13px] text-text-muted font-mono whitespace-nowrap">
                  {{ formatTransition(pod.lastTransition) }}
                </td>
              </tr>

              <tr
                v-if="pod.message"
                :class="[
                  'border-b border-surface-border-subtle',
                  pod.phase === 'Failed' ? 'bg-status-danger-bg/30' : '',
                ]"
              >
                <td colspan="6" class="px-3 pb-2 pt-0">
                  <div class="pl-0 lg:pl-[42%]">
                    <div class="rounded bg-surface-sunken px-2 py-1.5 text-[12px] leading-relaxed text-text-secondary">
                      <span class="mr-2 font-semibold uppercase tracking-wider text-text-muted">Diagnostic</span>
                      <span class="font-mono whitespace-pre-wrap break-words">{{ pod.message }}</span>
                    </div>
                  </div>
                </td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>
