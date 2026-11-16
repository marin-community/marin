<script setup>
import { computed } from "vue";
const props = defineProps({ value: null, depth: { type: Number, default: 0 } });
const decoded = computed(() => {
  if (typeof props.value === "string" && /^\s*[\[{]/.test(props.value)) {
    try {
      return JSON.parse(props.value);
    } catch {
      /* Bracketed text is also valid. */
    }
  }
  return props.value;
});
const nested = computed(
  () => decoded.value !== null && typeof decoded.value === "object",
);
</script>
<template>
  <details v-if="nested" :open="depth < 2" class="value-tree">
    <summary>
      {{
        Array.isArray(decoded)
          ? `Array · ${decoded.length} items`
          : `Object · ${Object.keys(decoded).length} fields`
      }}
    </summary>
    <div v-for="(item, key) in decoded" :key="key" class="tree-item">
      <span class="tree-key">{{ key }}</span
      ><ValueView :value="item" :depth="depth + 1" />
    </div>
  </details>
  <pre v-else :class="{ null: decoded === null }">{{
    decoded === null ? "null" : String(decoded)
  }}</pre>
</template>
