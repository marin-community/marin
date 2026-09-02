<script setup lang="ts">
// A task by name, anywhere in the dataset.
//
// The dataset server's full-text index answers this. It builds the index the
// first time anybody asks and says so until it is done; that message is shown
// as it arrives, and a retry is one press away.
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { corpus, count, sourceOf, type Corpus } from '../corpus'
import { PAGE, search, type Row } from '../trove'
import { sourcePath, taskPath } from '../routes'
import Verdict from '../components/Verdict.vue'

const route = useRoute()
const router = useRouter()
const loaded = ref<Corpus>()
const typed = ref(typeof route.query.q === 'string' ? route.query.q : '')
const found = ref<Row[]>([])
const total = ref(0)
const problem = ref('')
const searching = ref(false)
const asked = ref('')

const rows = computed(() =>
  found.value.map((row) => ({
    ...row,
    source: loaded.value ? sourceOf(loaded.value.sources, row.row) : undefined,
    label: loaded.value?.labelled.get(row.row),
  })),
)

async function run(): Promise<void> {
  const needle = typed.value.trim()
  if (!needle) return
  router.replace({ query: { q: needle } })
  searching.value = true
  problem.value = ''
  asked.value = needle
  try {
    const answer = await search(needle, 0)
    found.value = answer.rows
    total.value = answer.total
  } catch (error) {
    problem.value = String(error)
    found.value = []
    total.value = 0
  } finally {
    searching.value = false
  }
}

async function more(): Promise<void> {
  searching.value = true
  try {
    const answer = await search(asked.value, found.value.length)
    found.value = [...found.value, ...answer.rows]
  } catch (error) {
    problem.value = String(error)
  } finally {
    searching.value = false
  }
}

onMounted(async () => {
  loaded.value = await corpus()
  if (typed.value) await run()
})
watch(() => route.query.q, (q) => {
  if (typeof q === 'string' && q !== asked.value) {
    typed.value = q
    run()
  }
})
</script>

<template>
  <p class="note">
    Search every task name in the dataset. Names are what the generators wrote:
    <code>stack-cpp-8997</code>, <code>task_714</code>, <code>swesmith-…</code>.
  </p>
  <form class="controls" @submit.prevent="run">
    <input type="search" v-model="typed" placeholder="Part of a task name" autofocus />
    <button class="button" type="submit" :disabled="searching">{{ searching ? 'Searching…' : 'Search' }}</button>
    <span class="tally" v-if="asked && !problem">{{ count(total) }} matches for “{{ asked }}”</span>
  </form>
  <p class="problem" v-if="problem">{{ problem }} <button class="button quiet" type="button" @click="run">Try again</button></p>
  <div class="scroll" v-if="rows.length">
    <table>
      <thead>
        <tr><th>Task</th><th>Source</th><th>Row</th><th>Audit</th></tr>
      </thead>
      <tbody>
        <tr v-for="item in rows" :key="item.row">
          <td><RouterLink :to="taskPath(item.row)">{{ item.path }}</RouterLink></td>
          <td><RouterLink v-if="item.source" :to="sourcePath(item.source.source)">{{ item.source.source }}</RouterLink></td>
          <td class="num">{{ count(item.row) }}</td>
          <td><Verdict v-if="item.label" :value="item.label.shellsim_now" prefix="shellsim: " /></td>
        </tr>
      </tbody>
    </table>
  </div>
  <p class="empty" v-else-if="asked && !searching && !problem">Nothing is called that.</p>
  <div class="more" v-if="found.length < total">
    <button class="button" type="button" @click="more" :disabled="searching">Load {{ Math.min(PAGE, total - found.length) }} more</button>
  </div>
</template>
