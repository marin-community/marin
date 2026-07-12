<script setup lang="ts">
import { computed } from 'vue'
import { fmt } from '../utils/formatting'

export interface Bar {
  label: string
  count: number
}

const props = defineProps<{ bars: Bar[]; clickable?: boolean; selected?: number | null }>()
const emit = defineEmits<{ select: [number] }>()

const max = computed(() => Math.max(1, ...props.bars.map((b) => b.count)))
</script>

<template>
  <div>
    <div
      v-for="(bar, i) in bars"
      :key="i"
      class="my-px flex items-center gap-2 text-xs"
      :class="{ 'cursor-pointer hover:bg-surface-raised': clickable, 'font-bold': selected === i }"
      @click="clickable && emit('select', i)"
    >
      <span class="w-[150px] text-right font-mono text-text-muted">{{ bar.label }}</span>
      <span class="h-4 rounded-sm bg-accent" :style="{ width: `${Math.round((260 * bar.count) / max)}px` }"></span>
      <span class="text-text-muted">{{ fmt(bar.count) }}</span>
    </div>
  </div>
</template>
