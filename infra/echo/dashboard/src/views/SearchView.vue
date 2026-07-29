<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  fetchJson,
  formatDate,
  type FederatedResult,
  type RepositoryIndexStatus,
  type SearchDomain,
} from '../types'

const PAGE_SIZE = 30
const DEBOUNCE_MS = 250
const DISPLAY_SHA_CHARACTERS = 12
const DOMAINS: { value: SearchDomain; label: string }[] = [
  { value: 'file', label: 'Files' },
  { value: 'wiki', label: 'Wiki' },
  { value: 'pr', label: 'Pull requests' },
  { value: 'issue', label: 'Issues' },
  { value: 'discord', label: 'Discord' },
]
const DEFAULT_DOMAINS: SearchDomain[] = ['file', 'wiki', 'pr', 'issue']

const route = useRoute()
const router = useRouter()
const query = ref(typeof route.query.q === 'string' ? route.query.q : '')
const selectedDomains = ref<SearchDomain[]>(domainsFromQuery(route.query.domain))
const results = ref<FederatedResult[]>([])
const index = ref<RepositoryIndexStatus | null>(null)
const indexError = ref('')
const loading = ref(false)
const error = ref('')
let request: AbortController | null = null

function domainsFromQuery(value: unknown): SearchDomain[] {
  const requested = Array.isArray(value) ? value : typeof value === 'string' ? [value] : []
  const valid = requested.filter((domain): domain is SearchDomain =>
    DOMAINS.some((candidate) => candidate.value === domain),
  )
  return valid.length ? [...new Set(valid)] : DEFAULT_DOMAINS
}

const indexPercent = computed(() => {
  if (index.value?.status !== 'building' || !index.value.total_files) return 0
  return Math.round(((index.value.completed_files || 0) / index.value.total_files) * 100)
})

const resultLabel = computed(() => {
  if (loading.value) return 'Searching…'
  if (!query.value.trim()) return 'Search files, knowledge, and conversations'
  if (!results.value.length) return 'No matches'
  return `${results.value.length} ${results.value.length === 1 ? 'result' : 'results'}`
})

function toggleDomain(domain: SearchDomain): void {
  selectedDomains.value = selectedDomains.value.includes(domain)
    ? selectedDomains.value.filter((candidate) => candidate !== domain)
    : [...selectedDomains.value, domain]
}

function wikiPath(result: FederatedResult): string {
  return new URL(result.url).pathname
}

function indexLabel(): string {
  if (!index.value || index.value.status === 'empty') return 'The repository index has not started.'
  if (index.value.status === 'building') {
    return `Indexing ${index.value.completed_files || 0} of ${index.value.total_files || 0} files from ${
      index.value.branch
    }@${index.value.commit_sha?.slice(0, DISPLAY_SHA_CHARACTERS)}. Partial results are available.`
  }
  return `Files reflect ${index.value.branch}@${index.value.commit_sha?.slice(
    0,
    DISPLAY_SHA_CHARACTERS,
  )}, indexed ${formatDate(index.value.indexed_at)}.`
}

async function loadIndex(): Promise<void> {
  indexError.value = ''
  try {
    index.value = await fetchJson<RepositoryIndexStatus>('/api/repository-index')
  } catch (reason) {
    index.value = null
    indexError.value = reason instanceof Error ? reason.message : 'Index status unavailable'
  }
}

async function search(): Promise<void> {
  request?.abort()
  results.value = []
  const text = query.value.trim()
  if (!text || !selectedDomains.value.length) {
    loading.value = false
    return
  }

  request = new AbortController()
  loading.value = true
  error.value = ''
  const params = new URLSearchParams({ q: text, limit: String(PAGE_SIZE) })
  for (const domain of selectedDomains.value) params.append('domain', domain)
  try {
    results.value = await fetchJson<FederatedResult[]>(`/api/federated-search?${params}`, request.signal)
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === 'AbortError') return
    error.value = reason instanceof Error ? reason.message : 'Search failed'
  } finally {
    loading.value = false
  }
}

function syncUrl(): void {
  const params: Record<string, string | string[]> = {}
  if (query.value.trim()) params.q = query.value.trim()
  if (
    selectedDomains.value.length !== DEFAULT_DOMAINS.length ||
    selectedDomains.value.some((domain) => !DEFAULT_DOMAINS.includes(domain))
  ) {
    params.domain = selectedDomains.value
  }
  router.replace({ query: params }).catch(() => {})
}

function run(): void {
  syncUrl()
  search()
}

let debounceTimer: ReturnType<typeof setTimeout> | undefined
function debouncedRun(): void {
  clearTimeout(debounceTimer)
  debounceTimer = setTimeout(run, DEBOUNCE_MS)
}

watch(query, debouncedRun)
watch(selectedDomains, run, { deep: true })

function submit(): void {
  clearTimeout(debounceTimer)
  run()
}

onMounted(() => {
  loadIndex()
  search()
})
</script>

<template>
  <section class="max-w-4xl">
    <p class="font-mono text-xs uppercase tracking-[0.18em] text-fern">Wiki · code · GitHub · Discord</p>
    <h1 class="mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-5xl">Search across Marin.</h1>

    <div v-if="index" class="mt-6 border-l-2 border-fern/40 pl-4 text-sm text-ink/60">
      <div class="flex items-center justify-between gap-4">
        <p>{{ indexLabel() }}</p>
        <span v-if="index.status === 'building'" class="font-mono text-xs">{{ indexPercent }}%</span>
      </div>
      <div v-if="index.status === 'building'" class="mt-2 h-1.5 overflow-hidden rounded bg-line/60">
        <div class="h-full bg-fern transition-all" :style="{ width: `${indexPercent}%` }" />
      </div>
    </div>
    <p v-else-if="indexError" class="mt-6 border-l-2 border-amber-500 pl-4 text-sm text-ink/60">
      Repository index status unavailable: {{ indexError }}
    </p>
  </section>

  <form class="mt-8 max-w-4xl border-y border-line py-4" @submit.prevent="submit">
    <div class="flex flex-col gap-2 sm:flex-row">
      <label class="sr-only" for="echo-query">Search Echo</label>
      <input
        id="echo-query"
        v-model="query"
        class="min-w-0 flex-1 rounded-lg border border-line bg-white px-4 py-3 placeholder:text-ink/35"
        placeholder="Identifier, incident, question, or phrase…"
        type="search"
      />
      <button
        class="rounded-lg bg-moss px-6 py-3 font-semibold text-white hover:bg-fern disabled:cursor-wait disabled:opacity-60"
        :disabled="loading || !selectedDomains.length"
        type="submit"
      >
        {{ loading ? 'Searching' : 'Search' }}
      </button>
    </div>

    <fieldset class="mt-3 flex flex-wrap gap-x-4 gap-y-2">
      <legend class="sr-only">Search domains</legend>
      <label
        v-for="domain in DOMAINS"
        :key="domain.value"
        class="flex cursor-pointer items-center gap-2 text-sm text-ink/65"
      >
        <input
          class="size-4 accent-fern"
          type="checkbox"
          :checked="selectedDomains.includes(domain.value)"
          @change="toggleDomain(domain.value)"
        />
        {{ domain.label }}
      </label>
    </fieldset>
  </form>

  <section class="mt-7 max-w-4xl" aria-live="polite">
    <div class="mb-2 flex items-center justify-between">
      <h2 class="text-sm font-semibold text-ink/55">{{ resultLabel }}</h2>
      <span v-if="query.trim()" class="font-mono text-xs text-ink/35">hybrid ranked</span>
    </div>

    <div v-if="error" class="my-4 border-l-2 border-red-500 bg-red-50 px-4 py-3 text-sm text-red-800">
      {{ error }}
    </div>

    <div v-if="loading" class="divide-y divide-line border-t border-line">
      <div v-for="item in 4" :key="item" class="h-28 animate-pulse bg-white/35" />
    </div>

    <div v-else class="divide-y divide-line border-t border-line">
      <article v-for="result in results" :key="result.id" class="group py-4">
        <div class="flex items-start gap-3">
          <span class="mt-0.5 w-14 shrink-0 font-mono text-[11px] uppercase tracking-wide text-moss">
            {{ result.domain }}
          </span>
          <div class="min-w-0">
            <router-link
              v-if="result.domain === 'wiki'"
              class="font-semibold leading-snug group-hover:text-moss"
              :to="wikiPath(result)"
            >
              {{ result.title }}
            </router-link>
            <a
              v-else
              class="font-semibold leading-snug group-hover:text-moss"
              :href="result.url"
              rel="noreferrer"
              target="_blank"
            >
              {{ result.title }}
            </a>
            <p class="mt-1 break-words text-xs text-ink/45">{{ result.subtitle }}</p>
            <p class="mt-2 line-clamp-3 text-sm leading-6 text-ink/65">{{ result.snippet }}</p>
          </div>
        </div>
      </article>
    </div>
  </section>
</template>
