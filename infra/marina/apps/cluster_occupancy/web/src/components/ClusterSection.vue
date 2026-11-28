<script setup>
import { computed } from "vue";

import { fmtEastern, fmtUtc } from "../time.js";
import NodeTile from "./NodeTile.vue";

const props = defineProps({
  cluster: { type: Object, required: true },
  colorClass: { type: Function, required: true },
  hover: { type: Function, required: true },
});

// Nodes arrive sorted by rack then name, so grouping keeps that order.
const groups = computed(() => {
  const map = new Map();
  for (const node of props.cluster.nodes ?? []) {
    const key = node.rack || "";
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(node);
  }
  return [...map.entries()];
});

const noData = computed(() => {
  const reason = props.cluster.reason;
  if (props.cluster.state === "unavailable") {
    return { text: `The ${props.cluster.cluster} controller could not be reached.`, title: "" };
  }
  if (reason === "not_mirrored") {
    return { text: `No history — ${props.cluster.cluster} is not mirrored to the hub.`, title: "" };
  }
  if (reason === "before_coverage" && props.cluster.first_ms != null) {
    return {
      text: `No data before ${fmtEastern(props.cluster.first_ms)} — the mirror starts then.`,
      title: `Mirror coverage starts ${fmtUtc(props.cluster.first_ms)}`,
    };
  }
  if (reason === "after_coverage" && props.cluster.last_ms != null) {
    return {
      text: `No data after ${fmtEastern(props.cluster.last_ms)} — the mirror has not updated since.`,
      title: `Mirror coverage ends ${fmtUtc(props.cluster.last_ms)}`,
    };
  }
  return { text: "No data for this window.", title: "" };
});
</script>

<template>
  <section class="cluster" :aria-label="cluster.cluster">
    <h2>{{ cluster.cluster }}</h2>
    <div v-if="cluster.state === 'no_data' || cluster.state === 'unavailable'" class="no-data" :title="noData.title">
      <span>{{ noData.text }}</span>
    </div>
    <template v-else>
      <div v-for="[rack, nodes] in groups" :key="rack" class="rack">
        <h3 v-if="rack">{{ rack }}</h3>
        <div class="tiles">
          <NodeTile
            v-for="node in nodes"
            :key="node.node_name"
            :node="node"
            :color-class="colorClass"
            :hover="hover"
          />
        </div>
      </div>
    </template>
  </section>
</template>
