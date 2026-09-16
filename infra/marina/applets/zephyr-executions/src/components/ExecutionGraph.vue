<script setup>
import { computed, useId } from "vue";
import { stageLayout, STAGE_NODE_WIDTH, STAGE_NODE_HEIGHT } from "../zephyr.js";

const props = defineProps({
  stages: { type: Array, required: true },
  selectedStage: { type: String, required: true },
  statuses: { type: Object, required: true },
});
const emit = defineEmits(["select"]);
const markerId = useId();
const graph = computed(() => {
  try {
    return { layout: stageLayout(props.stages), error: "" };
  } catch (error) {
    return { layout: stageLayout([]), error: `Cannot draw stage dependencies: ${String(error)}` };
  }
});
</script>

<template>
  <p v-if="graph.error" role="alert" class="danger">{{ graph.error }}</p>
  <div v-else class="graph" aria-label="Stage dependencies">
    <div class="canvas" :style="{ width: `${graph.layout.width}px`, height: `${graph.layout.height}px` }">
      <svg :width="graph.layout.width" :height="graph.layout.height" aria-hidden="true">
        <defs>
          <marker :id="markerId" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0 0 L7 3.5 L0 7" fill="currentColor" />
          </marker>
        </defs>
        <path
          v-for="edge in graph.layout.edges"
          :key="edge.key"
          :d="`M${edge.x1},${edge.y1} C${edge.x1 + 16},${edge.y1} ${edge.x2 - 16},${edge.y2} ${edge.x2},${edge.y2}`"
          fill="none"
          stroke="currentColor"
          stroke-width="1.5"
          :marker-end="`url(#${markerId})`"
        />
      </svg>
      <button
        v-for="node in graph.layout.nodes"
        :key="node.stage_name"
        class="node"
        :title="node.stage_name"
        :aria-pressed="selectedStage === node.stage_name"
        :style="{ left: `${node.x}px`, top: `${node.y}px`, width: `${STAGE_NODE_WIDTH}px`, height: `${STAGE_NODE_HEIGHT}px` }"
        @click="emit('select', node)"
      >
        <span class="faint mono">{{ node.stage_name }}</span>
        <span class="label"><strong>{{ node.label }}</strong></span>
        <span class="status" :class="{ failed: statuses[node.stage_name] === 'Failed' }">{{ statuses[node.stage_name] }}</span>
      </button>
    </div>
  </div>
</template>

