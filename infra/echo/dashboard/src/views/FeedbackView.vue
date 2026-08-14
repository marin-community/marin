<script setup lang="ts">
import { ref } from 'vue'
import { fetchJson, formatDateTime, type SearchFeedbackEntry } from '../types'

const entries = ref<SearchFeedbackEntry[]>([])
const loading = ref(true)
const error = ref('')

function gradeClass(grade: number): string {
  if (grade >= 8) return 'bg-emerald-100 text-emerald-800'
  if (grade >= 4) return 'bg-amber-100 text-amber-800'
  return 'bg-red-100 text-red-800'
}

async function load(): Promise<void> {
  try {
    entries.value = await fetchJson<SearchFeedbackEntry[]>('/api/feedback?days=30&limit=200')
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : 'Could not load search feedback'
  } finally {
    loading.value = false
  }
}

load()
</script>

<template>
  <section class="max-w-3xl">
    <p class="font-mono text-xs uppercase tracking-[0.18em] text-fern">Search quality</p>
    <h1 class="mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-5xl">Feedback from agents.</h1>
    <p class="mt-4 max-w-2xl text-sm leading-6 text-ink/55">
      Result-level grades and concise notes from recent Echo searches. Ten means directly useful; zero means irrelevant.
    </p>
  </section>

  <section class="mt-8" aria-live="polite">
    <div v-if="error" class="border-l-2 border-red-500 bg-red-50 px-4 py-3 text-sm text-red-800">
      {{ error }}
    </div>

    <div v-if="loading" class="divide-y divide-line border-y border-line">
      <div v-for="index in 5" :key="index" class="h-20 animate-pulse bg-white/35" />
    </div>

    <div v-else-if="!entries.length && !error" class="border-y border-line py-12 text-center text-sm text-ink/45">
      No feedback in the last 30 days.
    </div>

    <template v-else>
      <div class="divide-y divide-line border-y border-line md:hidden">
        <article v-for="entry in entries" :key="entry.id" class="py-5">
          <p class="font-medium leading-5 text-ink/80">{{ entry.query }}</p>
          <p class="mt-1 leading-5 text-ink/50">{{ entry.note }}</p>
          <div v-if="entry.grades.length" class="mt-4 divide-y divide-line/70 border-y border-line/70">
            <div v-for="result in entry.grades" :key="result.result_id" class="flex items-start gap-3 py-3">
              <span
                class="inline-flex min-w-9 shrink-0 justify-center rounded-full px-2 py-1 font-mono text-xs font-semibold"
                :class="gradeClass(result.grade)"
              >
                {{ result.grade }}
              </span>
              <div class="min-w-0 flex-1">
                <a class="font-semibold leading-5 hover:text-moss" :href="result.url" rel="noreferrer" target="_blank">
                  {{ result.title }}
                </a>
                <span class="mt-1 block truncate font-mono text-[11px] text-ink/35">{{ result.result_id }}</span>
              </div>
            </div>
          </div>
          <p v-else class="mt-4 border-y border-line/70 py-3 text-sm text-ink/45">No individual results graded.</p>
          <p class="mt-3 text-xs text-ink/40">{{ entry.author }} · {{ formatDateTime(entry.created_at) }}</p>
        </article>
      </div>

      <div class="hidden overflow-x-auto border-y border-line md:block">
        <table class="w-full min-w-[860px] border-collapse text-left text-sm">
          <thead class="font-mono text-[11px] uppercase tracking-wide text-ink/40">
            <tr>
              <th class="w-20 px-3 py-3 font-medium">Grade</th>
              <th class="w-64 px-3 py-3 font-medium">Item</th>
              <th class="px-3 py-3 font-medium">Query and note</th>
              <th class="w-44 px-3 py-3 font-medium">Agent</th>
              <th class="w-40 px-3 py-3 font-medium">When</th>
            </tr>
          </thead>
          <tbody v-for="entry in entries" :key="entry.id" class="border-t border-line first:border-t-0">
            <tr v-if="!entry.grades.length" class="align-top hover:bg-white/35">
              <td class="px-3 py-4 font-mono text-ink/30">—</td>
              <td class="px-3 py-4 font-medium text-ink/55">Overall result set</td>
              <td class="px-3 py-4">
                <p class="font-medium leading-5 text-ink/80">{{ entry.query }}</p>
                <p class="mt-1 line-clamp-2 leading-5 text-ink/50">{{ entry.note }}</p>
              </td>
              <td class="max-w-44 truncate px-3 py-4 text-xs text-ink/50">{{ entry.author }}</td>
              <td class="px-3 py-4 text-xs text-ink/45">{{ formatDateTime(entry.created_at) }}</td>
            </tr>
            <tr v-for="(result, index) in entry.grades" v-else :key="result.result_id" class="align-top hover:bg-white/35">
              <td class="px-3 py-4">
                <span
                  class="inline-flex min-w-9 justify-center rounded-full px-2 py-1 font-mono text-xs font-semibold"
                  :class="gradeClass(result.grade)"
                >
                  {{ result.grade }}
                </span>
              </td>
              <td class="px-3 py-4">
                <a class="line-clamp-2 font-semibold leading-5 hover:text-moss" :href="result.url" rel="noreferrer" target="_blank">
                  {{ result.title }}
                </a>
                <span class="mt-1 block truncate font-mono text-[11px] text-ink/35">{{ result.result_id }}</span>
              </td>
              <td v-if="index === 0" class="px-3 py-4" :rowspan="entry.grades.length">
                <p class="font-medium leading-5 text-ink/80">{{ entry.query }}</p>
                <p class="mt-1 line-clamp-2 leading-5 text-ink/50">{{ entry.note }}</p>
              </td>
              <td v-if="index === 0" class="max-w-44 truncate px-3 py-4 text-xs text-ink/50" :rowspan="entry.grades.length">
                {{ entry.author }}
              </td>
              <td v-if="index === 0" class="px-3 py-4 text-xs text-ink/45" :rowspan="entry.grades.length">
                {{ formatDateTime(entry.created_at) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </section>
</template>
