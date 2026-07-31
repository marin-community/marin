<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { fetchJson, formatDate, type ActivityDetail } from '../types'

const props = defineProps<{ id: string }>()

const chunk = ref<Omit<ActivityDetail, 'type'> | null>(null)
const loading = ref(true)
const error = ref('')

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  chunk.value = null
  try {
    chunk.value = await fetchJson<Omit<ActivityDetail, 'type'>>(`/api/chunks/${props.id}`)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not open result'
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(() => props.id, load)
</script>

<template>
  <div v-if="loading" class="max-w-3xl">
    <div class="h-64 animate-pulse rounded-3xl border border-line bg-white/60" />
  </div>

  <div v-else-if="error" class="max-w-3xl rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
    {{ error }}
  </div>

  <article v-else-if="chunk" class="max-w-3xl rounded-3xl border border-line bg-white/90 p-6 shadow-card sm:p-9">
    <p class="font-mono text-xs uppercase tracking-widest text-fern">{{ chunk.source }} · {{ chunk.kind }}</p>
    <h1 class="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{{ chunk.title || 'Untitled' }}</h1>
    <p class="mt-3 text-sm text-ink/45">{{ chunk.author || 'unknown' }} · {{ formatDate(chunk.date) }}</p>
    <div class="mt-7 whitespace-pre-wrap text-[15px] leading-7 text-ink/80">{{ chunk.text || chunk.snippet }}</div>
    <div class="mt-8 flex items-center justify-end border-t border-line pt-5">
      <a
        class="rounded-lg border border-moss/25 bg-white px-4 py-2 text-sm font-semibold text-moss hover:bg-mist"
        :href="chunk.url"
        target="_blank"
        rel="noreferrer"
      >
        View original ↗
      </a>
    </div>
  </article>
</template>
