<script setup lang="ts">
// Every source in the dataset, with what the audit said about it.
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { corpus, count, type Corpus } from '../corpus'
import { sourcePath } from '../routes'
import Verdict from '../components/Verdict.vue'

const loaded = ref<Corpus>()
const problem = ref('')
const needle = ref('')
const verdict = ref('')
const unlock = ref('')

onMounted(async () => {
  try {
    loaded.value = await corpus()
  } catch (error) {
    problem.value = String(error)
  }
})

const unlocks = computed(() => {
  if (!loaded.value) return []
  const seen = new Set<string>()
  for (const review of loaded.value.reviews.values()) seen.add(review.cheapest_unlock)
  return [...seen].sort()
})

const rows = computed(() => {
  if (!loaded.value) return []
  const { sources, reviews, labels } = loaded.value
  const sampled = new Map<string, number>()
  for (const label of labels) sampled.set(label.source, (sampled.get(label.source) ?? 0) + 1)
  const wanted = needle.value.trim().toLowerCase()
  return sources
    .map((source) => ({ source, review: reviews.get(source.source), sampled: sampled.get(source.source) ?? 0 }))
    .filter(({ source, review }) => {
      if (wanted && !source.source.toLowerCase().includes(wanted) && !review?.template_description.toLowerCase().includes(wanted)) return false
      if (verdict.value && review?.shellsim_verdict !== verdict.value) return false
      if (unlock.value && review?.cheapest_unlock !== unlock.value) return false
      return true
    })
})

const shown = computed(() => rows.value.reduce((sum, row) => sum + row.source.rows, 0))

/** The first sentence of a review, for a table cell. */
function gist(text: string | undefined): string {
  if (!text) return ''
  const end = text.search(/\.\s/)
  return end < 0 ? text : text.slice(0, end + 1)
}
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <p class="note">
    Ninety-six sources, one parquet file each. A source is a task template and a
    generator; the audit read about eleven tasks from each and reviewed the
    source as a whole. Open a source to page through its tasks.
  </p>
  <div class="tally-row" v-if="loaded">
    <div class="stat"><b>{{ count(loaded.total) }}</b><span>tasks</span></div>
    <div class="stat"><b>{{ loaded.sources.length }}</b><span>sources</span></div>
    <div class="stat"><b>{{ count(loaded.labels.length) }}</b><span>audited</span></div>
  </div>
  <div class="controls">
    <input type="search" v-model="needle" placeholder="Filter sources by name or description" />
    <select v-model="verdict" aria-label="shellsim verdict">
      <option value="">any shellsim verdict</option>
      <option value="yes">runs today</option>
      <option value="partial">partial</option>
      <option value="no">does not run</option>
    </select>
    <select v-model="unlock" aria-label="cheapest unlock">
      <option value="">any unlock</option>
      <option v-for="u in unlocks" :key="u" :value="u">{{ u }}</option>
    </select>
    <span class="tally" v-if="loaded">{{ rows.length }} sources, {{ count(shown) }} tasks</span>
  </div>
  <div class="scroll" v-if="loaded">
    <table>
      <thead>
        <tr>
          <th>Source</th>
          <th>Tasks</th>
          <th>Audited</th>
          <th>shellsim</th>
          <th>Unlock</th>
          <th>What it is</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="{ source, review, sampled } in rows" :key="source.source">
          <td><RouterLink :to="sourcePath(source.source)">{{ source.source }}</RouterLink></td>
          <td class="num">{{ count(source.rows) }}</td>
          <td class="num">{{ sampled }}</td>
          <td><Verdict :value="review?.shellsim_verdict" /></td>
          <td><span class="chip" v-if="review && review.cheapest_unlock !== 'none'">{{ review.cheapest_unlock }}</span></td>
          <td>{{ gist(review?.template_description) }}</td>
        </tr>
      </tbody>
    </table>
    <p class="empty" v-if="rows.length === 0">No source matches that filter.</p>
  </div>
  <p class="working" v-else-if="!problem">Reading the manifest…</p>
</template>
