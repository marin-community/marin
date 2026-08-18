<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import MarkdownBody from '../MarkdownBody.vue'
import { fetchJson, formatDateTime, type WorkLogEntry, type WorkLogSummary } from '../types'

const LOOKBACK_DAYS = 30

const entries = ref<WorkLogSummary[]>([])
const details = reactive(new Map<number, WorkLogEntry>())
const detailErrors = reactive(new Map<number, string>())
const loadingDetails = reactive(new Set<number>())
const openEntry = ref<number | null>(null)
const loading = ref(true)
const error = ref('')

async function load(): Promise<void> {
  try {
    entries.value = await fetchJson<WorkLogSummary[]>(`/api/work_log?days=${LOOKBACK_DAYS}&limit=100`)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not load the conversation'
  } finally {
    loading.value = false
  }
}

async function toggle(entry: WorkLogSummary): Promise<void> {
  if (openEntry.value === entry.id) {
    openEntry.value = null
    return
  }
  openEntry.value = entry.id
  if (details.has(entry.id)) return

  loadingDetails.add(entry.id)
  detailErrors.delete(entry.id)
  try {
    details.set(entry.id, await fetchJson<WorkLogEntry>(`/api/work_log/${entry.id}`))
  } catch (reason) {
    detailErrors.set(entry.id, reason instanceof Error ? reason.message : 'Could not load details')
  } finally {
    loadingDetails.delete(entry.id)
  }
}

onMounted(load)
</script>

<template>
  <section class="max-w-3xl">
    <p class="font-mono text-xs uppercase tracking-[0.18em] text-fern">Shared work log</p>
    <h1 class="mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-5xl">What agents are working on.</h1>
    <p class="mt-4 max-w-2xl text-sm leading-6 text-ink/55">
      Recent milestones and active threads from Echo's shared agent work log.
    </p>
  </section>

  <section class="mt-8 max-w-4xl" aria-live="polite">
    <div v-if="error" class="border-l-2 border-red-500 bg-red-50 px-4 py-3 text-sm text-red-800">
      {{ error }}
    </div>

    <div v-if="loading" class="divide-y divide-line border-y border-line">
      <div v-for="index in 4" :key="index" class="h-24 animate-pulse bg-white/35" />
    </div>

    <div v-else-if="!entries.length && !error" class="border-y border-line py-12 text-center text-sm text-ink/45">
      No conversation entries in the last {{ LOOKBACK_DAYS }} days.
    </div>

    <div v-else class="divide-y divide-line border-y border-line">
      <article v-for="entry in entries" :key="entry.id">
        <button
          class="group flex w-full items-start gap-4 py-5 text-left"
          type="button"
          :aria-expanded="openEntry === entry.id"
          @click="toggle(entry)"
        >
          <span class="mt-1 hidden w-2 shrink-0 sm:block">
            <span class="block size-2 rounded-full bg-fern/60 group-hover:bg-fern" />
          </span>
          <span class="min-w-0 flex-1">
            <span class="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] uppercase tracking-wide text-ink/40">
              <span class="text-moss">{{ entry.project }}</span>
              <span>{{ formatDateTime(entry.at) }}</span>
            </span>
            <span class="mt-2 block text-base font-semibold leading-6 group-hover:text-moss">{{ entry.title }}</span>
            <span class="mt-1 block text-xs text-ink/40">{{ entry.author }}</span>
          </span>
          <span class="mt-1 text-lg text-ink/35" aria-hidden="true">{{ openEntry === entry.id ? '−' : '+' }}</span>
        </button>

        <div v-if="openEntry === entry.id" class="mb-5 ml-0 border-l-2 border-fern/25 pl-4 sm:ml-6">
          <p v-if="loadingDetails.has(entry.id)" class="py-2 text-sm text-ink/40">Loading details…</p>
          <p v-else-if="detailErrors.get(entry.id)" class="py-2 text-sm text-red-700">
            {{ detailErrors.get(entry.id) }}
          </p>
          <MarkdownBody v-else-if="details.get(entry.id)?.body" :source="details.get(entry.id)?.body || ''" />
          <p v-else class="py-2 text-sm text-ink/40">No additional details.</p>
        </div>
      </article>
    </div>
  </section>
</template>
