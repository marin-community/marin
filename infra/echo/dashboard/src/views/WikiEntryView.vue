<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { fetchJson, formatDate, type WikiHit } from '../types'
import MarkdownBody from '../MarkdownBody.vue'
import WikiTags from '../WikiTags.vue'

const props = defineProps<{ id: string }>()

const entry = ref<Omit<WikiHit, 'type'> | null>(null)
const loading = ref(true)
const error = ref('')

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  entry.value = null
  try {
    entry.value = await fetchJson<Omit<WikiHit, 'type'>>(`/api/wiki/${props.id}`)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not open note'
  } finally {
    loading.value = false
  }
}

async function markReferenced(): Promise<void> {
  if (!entry.value) return
  const response = await fetch(`/api/wiki/${entry.value.id}/references`, { method: 'POST' })
  if (!response.ok) {
    error.value = await response.text()
    return
  }
  entry.value = (await response.json()) as Omit<WikiHit, 'type'>
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

  <article v-else-if="entry" class="max-w-3xl">
    <p class="font-mono text-xs uppercase tracking-widest text-fern">Wiki note</p>
    <h1 class="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{{ entry.title }}</h1>
    <p class="mt-3 text-sm text-ink/45">{{ entry.author }} · {{ formatDate(entry.updated_at) }}</p>
    <p class="mt-6 border-l-2 border-fern/40 pl-4 text-sm font-medium leading-6 text-moss">
      Use when: {{ entry.use_when }}
    </p>
    <WikiTags :tags="entry.tags" class="mt-4" />
    <MarkdownBody class="mt-8" :source="entry.body || ''" />
    <div class="mt-8 flex items-center justify-between border-t border-line pt-5">
      <span class="text-sm text-ink/45">{{ entry.reference_count }} references</span>
      <button
        class="rounded border border-moss/25 bg-white px-4 py-2 text-sm font-semibold text-moss hover:bg-mist"
        @click="markReferenced"
      >
        Mark referenced
      </button>
    </div>
  </article>
</template>
