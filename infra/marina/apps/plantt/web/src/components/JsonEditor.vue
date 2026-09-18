<script setup>
import { ref, watch } from "vue";

import { resolvePlan } from "../model.js";

const props = defineProps({ plan: { type: Object, required: true } });
const emit = defineEmits(["apply", "error"]);

const text = ref(JSON.stringify(props.plan, null, 2));
const focused = ref(false);

watch(
  () => props.plan,
  (plan) => {
    if (!focused.value) text.value = JSON.stringify(plan, null, 2);
  },
  { deep: true },
);

function apply() {
  try {
    const parsed = JSON.parse(text.value);
    resolvePlan(parsed);
    emit("apply", parsed);
    emit("error", "");
  } catch (error) {
    emit("error", error instanceof Error ? error.message : String(error));
  }
}

function keydown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    apply();
  }
}
</script>

<template>
  <section class="json-editor">
    <div class="panel-heading">
      <div>
        <span class="eyebrow">Source of truth</span>
        <h2>Plan JSON</h2>
      </div>
      <button type="button" @click="apply">Apply</button>
    </div>
    <textarea
      v-model="text"
      aria-label="Plan JSON"
      spellcheck="false"
      @focus="focused = true"
      @blur="focused = false"
      @keydown="keydown"
    />
    <p class="editor-hint"><kbd>Ctrl</kbd>/<kbd>⌘</kbd> + <kbd>Enter</kbd> to apply</p>
  </section>
</template>
