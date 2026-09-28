<script setup>
import { computed, ref } from "vue";

const props = defineProps({
  charts: { type: Array, required: true },
  currentId: { type: String, default: "" },
  busy: { type: Boolean, default: false },
});

defineEmits(["open", "create-blank", "create-example", "import"]);

const query = ref("");
const shown = computed(() => {
  const needle = query.value.trim().toLowerCase();
  return needle ? props.charts.filter((chart) => chart.title.toLowerCase().includes(needle)) : props.charts;
});

function updatedAtLabel(value) {
  const elapsed = Date.now() - new Date(value).valueOf();
  if (elapsed < 60_000) return "just now";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
</script>

<template>
  <aside class="library" aria-label="Chart library">
    <div class="library-head">
      <div>
        <span class="eyebrow">Shared workspace</span>
        <h1>Plantt</h1>
      </div>
      <button class="icon-button" type="button" title="Import Plantt link or JSON" @click="$emit('import')">↥</button>
    </div>
    <div class="new-actions">
      <button type="button" :disabled="busy" @click="$emit('create-blank')">New plan</button>
      <button type="button" class="quiet-button" :disabled="busy" @click="$emit('create-example')">Use example</button>
    </div>
    <label class="search">
      <span class="sr-only">Filter charts</span>
      <input v-model="query" type="search" placeholder="Find a plan…" />
    </label>
    <div class="chart-list">
      <button
        v-for="chart in shown"
        :key="chart.id"
        type="button"
        class="chart-row"
        :aria-current="chart.id === currentId ? 'page' : undefined"
        @click="$emit('open', chart.id)"
      >
        <span class="chart-row-title">{{ chart.title }}</span>
        <span class="chart-row-meta">v{{ chart.revision }} · {{ updatedAtLabel(chart.updated_at) }}</span>
      </button>
      <p v-if="!shown.length" class="library-empty">{{ charts.length ? "No matching plans" : "No plans yet" }}</p>
    </div>
  </aside>
</template>
