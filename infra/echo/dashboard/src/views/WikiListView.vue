<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { fetchJson, formatDate, type WikiHit } from '../types'
import WikiTags from '../WikiTags.vue'

const PAGE_SIZE = 30

const entries = ref<Omit<WikiHit, 'type'>[]>([])
const loading = ref(true)
const error = ref('')

async function load(): Promise<void> {
  loading.value = true
  error.value = ''
  try {
    const params = new URLSearchParams({ q: '', limit: String(PAGE_SIZE) })
    entries.value = await fetchJson<Omit<WikiHit, 'type'>[]>(`/api/wiki/search?${params}`)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not load wiki notes'
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <section class="max-w-3xl">
    <p class="font-mono text-xs uppercase tracking-[0.18em] text-fern">Agent wiki</p>
    <h1 class="mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-5xl">Recently updated notes.</h1>
  </section>

  <section class="mt-8" aria-live="polite">
    <div v-if="error" class="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      {{ error }}
    </div>

    <div v-if="loading" class="grid gap-3">
      <div v-for="index in 3" :key="index" class="h-32 animate-pulse rounded-2xl border border-line bg-white/60" />
    </div>

    <div v-else class="divide-y divide-line border-t border-line">
      <article
        v-for="entry in entries"
        :key="entry.id"
        class="group py-5"
      >
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0">
            <div class="mb-2 flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-wide text-ink/45">
              <span class="text-moss">wiki</span>
              <span>{{ formatDate(entry.updated_at) }}</span>
            </div>
            <router-link
              class="text-left text-lg font-semibold leading-snug text-ink group-hover:text-moss"
              :to="`/wiki/${entry.id}`"
            >
              {{ entry.title }}
            </router-link>
            <p class="mt-2 text-sm font-medium leading-6 text-moss">Use when: {{ entry.use_when }}</p>
            <WikiTags :tags="entry.tags" class="mt-2" />
            <p class="mt-2 line-clamp-2 text-sm leading-6 text-ink/60">{{ entry.snippet }}</p>
          </div>
          <span class="shrink-0 text-xs text-ink/40">{{ entry.reference_count }} refs</span>
        </div>
        <p class="mt-3 text-xs text-ink/40">by {{ entry.author || 'unknown' }}</p>
      </article>
    </div>
  </section>
</template>
