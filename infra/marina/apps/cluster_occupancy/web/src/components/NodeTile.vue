<script setup>
import { computed } from "vue";

const props = defineProps({
  node: { type: Object, required: true },
  colorClass: { type: Function, required: true },
  hover: { type: Function, required: true },
});

// A holder's segment grows with the GPUs it holds; the hatched remainder is
// the node's observed capacity sitting idle.
const capacity = computed(() => Math.max(props.node.capacity_gpus || 0, 1));
const held = computed(() => props.node.holders.reduce((sum, holder) => sum + holder.gpus, 0));
const idleGpus = computed(() => Math.max(capacity.value - held.value, 0));

function segmentLabel(holder) {
  const ended = holder.finished_ms ? "ended" : "running";
  return `${holder.user_id}: ${holder.gpus} GPUs on ${props.node.node_name}, ${ended}`;
}
</script>

<template>
  <div class="tile" :id="`node-${node.node_name}`" @mouseleave="hover(null)">
    <span class="node-name" :title="node.node_name">{{ node.node_name }}</span>
    <div
      class="bar"
      role="img"
      :aria-label="idleGpus === capacity ? `${node.node_name}: idle, ${capacity} GPUs` : `${node.node_name}: ${held} of ${capacity} GPUs held`"
    >
      <div
        v-for="holder in node.holders"
        :key="holder.job_id"
        class="segment"
        :class="colorClass(holder.user_id)"
        :style="{ flexGrow: holder.gpus }"
        tabindex="0"
        :aria-label="segmentLabel(holder)"
        @mouseenter="hover({ holder, node: node.node_name, x: $event.clientX, y: $event.clientY })"
        @focus="hover({ holder, node: node.node_name, x: 8, y: 8 })"
      ></div>
      <div
        v-if="idleGpus > 0"
        class="segment idle"
        :style="{ flexGrow: idleGpus }"
        tabindex="0"
        :aria-label="`${node.node_name}: ${idleGpus} GPUs idle`"
        @mouseenter="hover({ idle: true, node: node.node_name, x: $event.clientX, y: $event.clientY })"
        @focus="hover({ idle: true, node: node.node_name, x: 8, y: 8 })"
      ></div>
    </div>
  </div>
</template>
