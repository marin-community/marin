<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

type Scope = 'all' | 'activity' | 'wiki'
type Source = 'all' | 'github' | 'discord'

interface ActivityHit {
  type: 'activity'
  id: number
  source: string
  kind: string
  date: string | null
  author: string | null
  title: string | null
  url: string
  snippet: string
  score: number
}

interface WikiHit {
  type: 'wiki'
  id: number
  created_at: string
  updated_at: string
  author: string
  title: string
  snippet: string
  reference_count: number
  score: number
  body?: string
}

type Result = ActivityHit | WikiHit

const query = ref('')
const scope = ref<Scope>('all')
const source = ref<Source>('all')
const results = ref<Result[]>([])
const loading = ref(false)
const error = ref('')
const selectedWiki = ref<WikiHit | null>(null)
let request: AbortController | null = null

const resultLabel = computed(() => {
  if (loading.value) return 'Searching…'
  if (!results.value.length) {
    if (query.value.trim()) return 'No matches'
    return scope.value === 'activity' ? 'Enter a query to search activity' : 'Recently updated wiki notes'
  }
  return `${results.value.length} ${results.value.length === 1 ? 'result' : 'results'}`
})

function formatDate(value: string | null): string {
  if (!value) return 'Date unknown'
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(new Date(value))
}

function activityUrl(): string {
  const params = new URLSearchParams({ q: query.value.trim(), limit: '20' })
  if (source.value !== 'all') params.set('source', source.value)
  return `search?${params}`
}

function wikiUrl(): string {
  return `wiki/search?${new URLSearchParams({ q: query.value.trim(), limit: '20' })}`
}

async function fetchJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function search(): Promise<void> {
  request?.abort()
  request = new AbortController()
  loading.value = true
  error.value = ''
  selectedWiki.value = null
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

function resultTime(result: Result): number {
  const value = result.type === 'wiki' ? result.updated_at : result.date
  return value ? new Date(value).getTime() : 0
}

async function openWiki(hit: WikiHit): Promise<void> {
  error.value = ''
  try {
    const detail = await fetchJson<Omit<WikiHit, 'type'>>(`wiki/${hit.id}`, new AbortController().signal)
    selectedWiki.value = { ...detail, type: 'wiki' }
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not open note'
  }
}

async function markReferenced(): Promise<void> {
  if (!selectedWiki.value) return
  const response = await fetch(`wiki/${selectedWiki.value.id}/references`, { method: 'POST' })
  if (!response.ok) {
    error.value = await response.text()
    return
  }
  const updated = (await response.json()) as Omit<WikiHit, 'type'>
  selectedWiki.value = { ...updated, type: 'wiki' }
  const index = results.value.findIndex((result) => result.type === 'wiki' && result.id === updated.id)
  if (index >= 0) results.value[index] = { ...updated, type: 'wiki' }
}

onMounted(search)
</script>

<template>
  <div class="min-h-screen">
    <header class="border-b border-line/80 bg-cream/75 backdrop-blur">
      <div class="mx-auto flex max-w-6xl items-center justify-between px-5 py-5 sm:px-8">
        <div class="flex items-center gap-3">
          <div class="grid size-10 place-items-center rounded-xl bg-moss text-lg font-semibold text-white shadow-card">E</div>
          <div>
            <p class="text-lg font-semibold tracking-tight">Echo</p>
            <p class="text-xs text-ink/55">Marin’s shared memory</p>
          </div>
        </div>
        <a class="text-sm font-medium text-moss hover:text-fern" href="docs">API docs</a>
      </div>
    </header>

    <main class="mx-auto max-w-6xl px-5 pb-16 pt-12 sm:px-8 sm:pt-16">
      <section class="max-w-3xl">
        <p class="mb-3 font-mono text-xs uppercase tracking-[0.22em] text-fern">Search what the team knows</p>
        <h1 class="text-4xl font-semibold leading-tight tracking-[-0.035em] sm:text-6xl">
          Find the thread.<br />
          Keep the knowledge.
        </h1>
        <p class="mt-5 max-w-2xl text-base leading-7 text-ink/65 sm:text-lg">
          Search agent wiki notes alongside Marin’s GitHub and Discord activity.
        </p>
      </section>

      <form class="mt-10 rounded-2xl border border-line bg-white/90 p-3 shadow-card" @submit.prevent="search">
        <div class="flex flex-col gap-3 sm:flex-row">
          <label class="sr-only" for="echo-query">Search Echo</label>
          <input
            id="echo-query"
            v-model="query"
            class="min-w-0 flex-1 rounded-xl border-0 bg-mist/65 px-5 py-4 text-base placeholder:text-ink/35"
            placeholder="Try “grafana”, a run name, or a question…"
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
                <a
                  v-if="result.type === 'activity'"
                  class="text-left text-lg font-semibold leading-snug text-ink group-hover:text-moss"
                  :href="result.url"
                  target="_blank"
                  rel="noreferrer"
                >
                  {{ result.title || result.snippet }}
                </a>
                <button
                  v-else
                  class="text-left text-lg font-semibold leading-snug text-ink group-hover:text-moss"
                  type="button"
                  @click="openWiki(result)"
                >
                  {{ result.title }}
                </button>
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
    </main>

    <div
      v-if="selectedWiki"
      class="fixed inset-0 z-20 grid place-items-end bg-ink/35 p-0 backdrop-blur-sm sm:place-items-center sm:p-6"
      role="dialog"
      aria-modal="true"
      :aria-label="selectedWiki.title"
      @click.self="selectedWiki = null"
    >
      <article class="max-h-[92vh] w-full overflow-y-auto rounded-t-3xl bg-cream p-6 shadow-2xl sm:max-w-3xl sm:rounded-3xl sm:p-9">
        <div class="flex items-start justify-between gap-4">
          <div>
            <p class="font-mono text-xs uppercase tracking-widest text-fern">Wiki note</p>
            <h2 class="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">{{ selectedWiki.title }}</h2>
          </div>
          <button class="rounded-lg px-3 py-2 text-2xl leading-none text-ink/45 hover:bg-mist" @click="selectedWiki = null">
            ×
          </button>
        </div>
        <p class="mt-3 text-sm text-ink/45">
          {{ selectedWiki.author }} · {{ formatDate(selectedWiki.updated_at) }}
        </p>
        <div class="mt-7 whitespace-pre-wrap text-[15px] leading-7 text-ink/80">{{ selectedWiki.body }}</div>
        <div class="mt-8 flex items-center justify-between border-t border-line pt-5">
          <span class="text-sm text-ink/45">{{ selectedWiki.reference_count }} references</span>
          <button class="rounded-lg border border-moss/25 bg-white px-4 py-2 text-sm font-semibold text-moss hover:bg-mist" @click="markReferenced">
            Mark referenced
          </button>
        </div>
      </article>
    </div>
  </div>
</template>
