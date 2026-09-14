<script setup lang="ts">
import { computed } from 'vue'
import { tipState } from '@/composables/tooltip'

// Offset from the cursor, and a right/bottom clamp so the box never leaves the viewport.
const style = computed(() => {
  const w = 260
  const left = Math.min(tipState.x + 14, window.innerWidth - w - 8)
  return { left: `${Math.max(8, left)}px`, top: `${tipState.y + 18}px`, maxWidth: `${w}px` }
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="tipState.show && tipState.content"
      class="fixed z-[70] pointer-events-none rounded-lg px-3 py-2 text-xs font-mono shadow-lg"
      style="background: var(--c-text); color: var(--c-page)"
      :style="style"
    >
      <div class="font-semibold">{{ tipState.content.title }}</div>
      <div
        v-for="(l, i) in tipState.content.lines ?? []"
        :key="i"
        class="flex justify-between gap-4 mt-0.5"
        :class="{ 'opacity-60': l.tone === 'muted' }"
        :style="l.tone === 'best' ? { color: 'var(--c-best)' } : undefined"
      >
        <span>{{ l.label }}</span><span class="tabular-nums">{{ l.value }}</span>
      </div>
    </div>
  </Teleport>
</template>
