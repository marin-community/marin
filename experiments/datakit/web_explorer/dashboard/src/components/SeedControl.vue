<script setup lang="ts">
import { randomSeed } from '../utils/formatting'

const seed = defineModel<number>({ required: true })
const emit = defineEmits<{ reroll: [] }>()

function reroll() {
  seed.value = randomSeed()
  emit('reroll')
}
</script>

<template>
  <label class="flex items-center gap-1.5 text-sm text-text-secondary">
    seed
    <input
      v-model.number="seed"
      type="number"
      class="w-[74px] rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
    />
  </label>
  <button
    class="rounded-md border border-surface-border px-2.5 py-1.5 text-sm hover:bg-surface-raised"
    title="new random seed & reload"
    @click="reroll"
  >
    🎲
  </button>
</template>
