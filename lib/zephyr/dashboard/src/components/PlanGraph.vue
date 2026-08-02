<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import StateBadge from '@/components/StateBadge.vue'
import type { PipelinePlan, PipelineStatus, PlanNode } from '@/types/dashboard'

const props = defineProps<{ plan: PipelinePlan; status: PipelineStatus | null }>()

const selectedId = ref('')
const mainNodes = computed(() => (props.plan.nodes ?? []).filter((node) => !node.auxiliary))
const stateByNode = computed(() =>
  Object.fromEntries((props.status?.nodeStatuses ?? []).map((item) => [item.nodeId, item.state])),
)
const selected = computed(() => (props.plan.nodes ?? []).find((node) => node.nodeId === selectedId.value) ?? null)

function auxiliaryNodes(node: PlanNode): PlanNode[] {
  return (props.plan.nodes ?? []).filter((candidate) => candidate.auxiliary && candidate.parentNodeId === node.nodeId)
}

function select(node: PlanNode) {
  selectedId.value = node.nodeId
}

watch(
  () => props.status?.currentNodeId,
  (nodeId) => {
    if (!selectedId.value && nodeId) selectedId.value = nodeId
  },
  { immediate: true },
)
</script>

<template>
  <div class="grid min-h-[560px] gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
    <div class="card overflow-auto p-6">
      <div class="mx-auto flex w-full max-w-2xl flex-col items-center">
        <template v-for="(node, index) in mainNodes" :key="node.nodeId">
          <div v-if="index" class="h-8 w-px bg-surface-border" />
          <div class="relative w-full max-w-md">
            <button
              :class="[
                'w-full rounded-xl border bg-surface-raised px-5 py-4 text-left shadow-sm hover:border-accent',
                selectedId === node.nodeId ? 'border-accent ring-2 ring-accent/15' : 'border-surface-border',
                status?.currentNodeId === node.nodeId ? 'border-accent' : '',
              ]"
              @click="select(node)"
            >
              <div class="flex items-start gap-3">
                <div class="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent-subtle text-xs font-bold text-accent">
                  {{ node.stageType === 'SOURCE' ? 'IN' : (node.stageIndex ?? 0) + 1 }}
                </div>
                <div class="min-w-0 flex-1">
                  <div class="flex items-center justify-between gap-2">
                    <p class="truncate text-sm font-semibold">{{ node.label || node.stageType }}</p>
                    <StateBadge :value="stateByNode[node.nodeId]" prefix="PLAN_NODE_STATE_" />
                  </div>
                  <p class="mt-1 text-xs text-text-secondary">
                    {{ node.stageType }} · {{ node.outputShards ?? 0 }} output shards
                  </p>
                  <div v-if="node.operationTypes?.length" class="mt-2 flex flex-wrap gap-1">
                    <span
                      v-for="operation in node.operationTypes"
                      :key="operation"
                      class="rounded bg-surface-sunken px-2 py-0.5 font-mono text-[11px] text-text-secondary"
                    >
                      {{ operation }}
                    </span>
                  </div>
                </div>
              </div>
            </button>

            <div v-if="auxiliaryNodes(node).length" class="ml-8 mt-3 border-l-2 border-dashed border-accent/40 pl-4">
              <p class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">Join input</p>
              <button
                v-for="auxiliary in auxiliaryNodes(node)"
                :key="auxiliary.nodeId"
                class="mb-2 w-full rounded-lg border border-surface-border bg-surface-sunken px-3 py-2 text-left hover:border-accent"
                @click="select(auxiliary)"
              >
                <span class="text-xs font-medium">{{ auxiliary.label }}</span>
                <span class="ml-2 text-[10px] text-text-muted">{{ auxiliary.stageType }}</span>
              </button>
            </div>
          </div>
        </template>
      </div>
    </div>

    <aside class="card h-fit p-5">
      <template v-if="selected">
        <p class="text-[11px] font-semibold uppercase tracking-wider text-text-muted">Selected stage</p>
        <h2 class="mt-2 text-lg font-semibold">{{ selected.label }}</h2>
        <StateBadge class="mt-3" :value="stateByNode[selected.nodeId]" prefix="PLAN_NODE_STATE_" />
        <dl class="mt-5 space-y-3 text-sm">
          <div>
            <dt class="text-xs text-text-muted">Node ID</dt>
            <dd class="mt-1 break-all font-mono text-xs">{{ selected.nodeId }}</dd>
          </div>
          <div class="flex justify-between gap-4">
            <dt class="text-text-secondary">Stage type</dt>
            <dd class="font-medium">{{ selected.stageType }}</dd>
          </div>
          <div class="flex justify-between gap-4">
            <dt class="text-text-secondary">Output shards</dt>
            <dd class="font-medium">{{ selected.outputShards ?? 0 }}</dd>
          </div>
          <div class="flex justify-between gap-4">
            <dt class="text-text-secondary">Auxiliary plan</dt>
            <dd class="font-medium">{{ selected.auxiliary ? 'Yes' : 'No' }}</dd>
          </div>
        </dl>
      </template>
      <p v-else class="text-sm text-text-secondary">Select a stage to inspect it.</p>
    </aside>
  </div>
</template>
