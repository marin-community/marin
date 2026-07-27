<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { fetchJson, formatDate, type ActivityHit, type Result, type WikiHit } from '../types'

type Scope = 'all' | 'activity' | 'wiki'
type Source = 'all' | 'github' | 'discord'

const PAGE_SIZE = 20

const query = ref('')
const scope = ref<Scope>('all')
const source = ref<Source>('all')
const results = ref<Result[]>([])
const loading = ref(false)
const error = ref('')
let request: AbortController | null = null

const resultLabel = computed(() => {
  if (loading.value) return 'Searching…'
  if (!results.value.length) {
    if (query.value.trim()) return 'No matches'
    return scope.value === 'activity' ? 'Enter a query to search activity' : 'Recently updated wiki notes'
  }
  return `${results.value.length} ${results.value.length === 1 ? 'result' : 'results'}`
})

function activityUrl(): string {
  const params = new URLSearchParams({ q: query.value.trim(), limit: String(PAGE_SIZE) })
  if (source.value !== 'all') params.set('source', source.value)
  return `/api/search?${params}`
}

function wikiUrl(): string {
  return `/api/wiki/search?${new URLSearchParams({ q: query.value.trim(), limit: String(PAGE_SIZE) })}`
}

function resultTime(result: Result): number {
  const value = result.type === 'wiki' ? result.updated_at : result.date
  return value ? new Date(value).getTime() : 0
}

function resultLink(result: Result): string {
  return result.type === 'wiki' ? `/wiki/${result.id}` : `/chunk/${result.id}`
}

function resultTitle(result: Result): string {
  return result.type === 'wiki' ? result.title : result.title || result.snippet
}

async function search(): Promise<void> {
  request?.abort()
  request = new AbortController()
  loading.value = true
  error.value = ''
  const text = query.value.trim()

  try {
    const jobs: Promise<Result[]>[] = []
    if (text && scope.value !== 'wiki') {
      jobs.push(
        fetchJson<Omit<ActivityHit, 'type'>[]>(activityUrl(), request.signal).then((hits) =>
          hits.map((hit) => ({ ...hit, type: 'activity' as const })),
        ),
      )
    }
    if (scope.value !== 'activity') {
      jobs.push(
        fetchJson<Omit<WikiHit, 'type'>[]>(wikiUrl(), request.signal).then((hits) =>
          hits.map((hit) => ({ ...hit, type: 'wiki' as const })),
        ),
      )
    }
    const groups = await Promise.all(jobs)
    results.value = groups
      .flat()
      .sort((left, right) => right.score - left.score || resultTime(right) - resultTime(left))
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === 'AbortError') return
    error.value = reason instanceof Error ? reason.message : 'Search failed'
    results.value = []
  } finally {
    loading.value = false
  }
}

onMounted(search)
</script>

<template>
  <section class="max-w-3xl">
    <p class="mb-3 font-mono text-xs uppercase tracking-[0.22em] text-fern">GitHub · Discord · agent wiki</p>
    <h1 class="text-4xl font-semibold leading-tight tracking-[-0.035em] sm:text-6xl">
      Search Marin activity and wiki notes.
    </h1>
    <p class="mt-5 max-w-2xl text-base leading-7 text-ink/65 sm:text-lg">
      Hybrid lexical and semantic search over the GitHub and Discord corpus and the
      agent wiki. Every hit links to its source.
    </p>
  </section>

  <form class="mt-10 rounded-2xl border border-line bg-white/90 p-3 shadow-card" @submit.prevent="search">
    <div class="flex flex-col gap-3 sm:flex-row">
      <label class="sr-only" for="echo-query">Search Echo</label>
      <input
        id="echo-query"
        v-model="query"
        class="min-w-0 flex-1 rounded-xl border-0 bg-mist/65 px-5 py-4 text-base placeholder:text-ink/35"
        placeholder="An identifier, a run name, or a question…"
        type="search"
      />
      <button
        class="rounded-xl bg-moss px-7 py-4 font-semibold text-white transition hover:bg-fern disabled:cursor-wait disabled:opacity-60"
        :disabled="loading"
        type="submit"
      >
        {{ loading ? 'Searching' : 'Search' }}
      </button>
    </div>

    <div class="mt-3 flex flex-wrap items-center gap-2 px-1 pb-1">
      <div class="flex rounded-lg bg-mist p-1" aria-label="Search scope">
        <button
          v-for="option in (['all', 'activity', 'wiki'] as Scope[])"
          :key="option"
          class="rounded-md px-3 py-1.5 text-sm capitalize transition"
          :class="scope === option ? 'bg-white font-semibold shadow-sm' : 'text-ink/55 hover:text-ink'"
          type="button"
          @click="scope = option; search()"
        >
          {{ option }}
        </button>
      </div>
      <select
        v-model="source"
        class="rounded-lg border border-line bg-white px-3 py-2 text-sm text-ink/70 disabled:opacity-40"
        :disabled="scope === 'wiki'"
        aria-label="Activity source"
        @change="search"
      >
        <option value="all">GitHub + Discord</option>
        <option value="github">GitHub</option>
        <option value="discord">Discord</option>
      </select>
    </div>
  </form>

  <section class="mt-8" aria-live="polite">
    <div class="mb-4 flex items-center justify-between">
      <h2 class="text-sm font-semibold text-ink/60">{{ resultLabel }}</h2>
      <span v-if="query.trim()" class="font-mono text-xs text-ink/40">hybrid · lexical + semantic</span>
    </div>

    <div v-if="error" class="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      {{ error }}
    </div>

    <div v-if="loading" class="grid gap-3">
      <div v-for="index in 3" :key="index" class="h-32 animate-pulse rounded-2xl border border-line bg-white/60" />
    </div>

    <div v-else class="grid gap-3">
      <article
        v-for="result in results"
        :key="`${result.type}-${result.id}`"
        class="group rounded-2xl border border-line bg-white/85 p-5 transition hover:-translate-y-0.5 hover:border-fern/40 hover:shadow-card sm:p-6"
      >
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0">
            <div class="mb-2 flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-wide text-ink/45">
              <span class="rounded-md bg-mist px-2 py-1 text-moss">{{ result.type }}</span>
              <span v-if="result.type === 'activity'">{{ result.source }} · {{ result.kind }}</span>
              <span>{{ formatDate(result.type === 'wiki' ? result.updated_at : result.date) }}</span>
            </div>
            <router-link
              class="text-left text-lg font-semibold leading-snug text-ink group-hover:text-moss"
              :to="resultLink(result)"
            >
              {{ resultTitle(result) }}
            </router-link>
            <p v-if="result.type === 'wiki'" class="mt-2 text-sm font-medium leading-6 text-moss">
              Use when: {{ result.use_when }}
            </p>
            <p class="mt-2 line-clamp-2 text-sm leading-6 text-ink/60">{{ result.snippet }}</p>
          </div>
          <span v-if="result.type === 'wiki'" class="shrink-0 text-xs text-ink/40">
            {{ result.reference_count }} refs
          </span>
        </div>
        <p class="mt-4 text-xs text-ink/40">by {{ result.author || 'unknown' }}</p>
      </article>
    </div>
  </section>
</template>
