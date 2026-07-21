<script setup lang="ts">
defineProps<{ state: string }>()

function colors(state: string): string {
  if (state === 'investigating') return 'bg-[#d9ebff] text-[#17568f] border-[#b5d6f5]'
  if (state === 'sending') return 'bg-[#d9ebff] text-[#17568f] border-[#b5d6f5]'
  if (state === 'pending') return 'bg-[#fff1c7] text-[#7a5110] border-[#ead597]'
  if (state === 'waiting_human') return 'bg-[#e8e0ff] text-[#5f4098] border-[#cfc0ef]'
  if (state === 'failed' || state === 'abandoned') return 'bg-[#ffe0dd] text-[#9b3028] border-[#f2bdb8]'
  if (state === 'archived') return 'bg-[#e7e9e5] text-[#59615c] border-[#d2d6d1]'
  return 'bg-[#dff2df] text-[#2a6b35] border-[#bfddc0]'
}
</script>

<template>
  <span class="inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide" :class="colors(state)">
    {{ state.replace('_', ' ') }}
  </span>
</template>
