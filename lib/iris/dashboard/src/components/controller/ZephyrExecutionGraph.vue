<script setup lang="ts">
import { computed, useId } from 'vue'
import { stageLayout, STAGE_NODE_WIDTH, STAGE_NODE_HEIGHT, type ExecutionStage } from '@/utils/zephyr'

const props = defineProps<{
  stages: ExecutionStage[]
  selectedStage: string
  statuses: Record<string, string>
}>()
const emit = defineEmits<{ select: [stage: ExecutionStage] }>()
const markerId = useId()
const graph = computed(() => {
  try {
    return { layout: stageLayout(props.stages), error: '' }
  } catch (error) {
    return { layout: stageLayout([]), error: `Cannot draw stage dependencies: ${String(error)}` }
  }
})
</script>

<template>
  <p v-if="graph.error" role="alert" class="text-red-500">{{ graph.error }}</p>
  <div v-else class="overflow-x-auto rounded bg-surface-sunken p-2" aria-label="Stage dependencies">
    <div class="relative" :style="{ width: `${graph.layout.width}px`, height: `${graph.layout.height}px` }">
      <svg class="pointer-events-none absolute inset-0 text-text-muted" :width="graph.layout.width" :height="graph.layout.height" aria-hidden="true">
        <defs><marker :id="markerId" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="currentColor" /></marker></defs>
        <path v-for="edge in graph.layout.edges" :key="edge.key"
          :d="`M${edge.x1},${edge.y1} C${edge.x1 + 16},${edge.y1} ${edge.x2 - 16},${edge.y2} ${edge.x2},${edge.y2}`"
          fill="none" stroke="currentColor" stroke-width="1.5" :marker-end="`url(#${markerId})`" />
      </svg>
      <button v-for="node in graph.layout.nodes" :key="node.stage_name" :title="node.stage_name"
        :aria-pressed="selectedStage === node.stage_name" @click="emit('select', node)"
        class="absolute flex flex-col justify-center gap-1 rounded-lg border bg-surface px-3 text-left focus-visible:outline-2 focus-visible:outline-accent"
        :class="selectedStage === node.stage_name ? 'border-accent ring-1 ring-accent' : 'border-surface-border hover:border-text-muted'"
        :style="{ left: `${node.x}px`, top: `${node.y}px`, width: `${STAGE_NODE_WIDTH}px`, height: `${STAGE_NODE_HEIGHT}px` }">
        <span class="w-full truncate text-xs text-text-muted">{{ node.stage_name }}</span>
        <span class="w-full truncate font-semibold text-text">{{ node.label }}</span>
        <span class="text-xs" :class="statuses[node.stage_name] === 'Failed' ? 'text-red-500' : 'text-text-secondary'">{{ statuses[node.stage_name] }}</span>
      </button>
    </div>
  </div>
</template>
