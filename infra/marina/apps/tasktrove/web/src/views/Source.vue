<script setup lang="ts">
// One source: the audit's review of it, then its tasks a page at a time.
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { bytes, corpus, count, type Corpus, type Label, type Source } from '../corpus'
import { PAGE, rows as readRows, type Row } from '../trove'
import { taskPath } from '../routes'
import Verdict from '../components/Verdict.vue'
import Score from '../components/Score.vue'

const props = defineProps<{ source: string }>()

const loaded = ref<Corpus>()
const problem = ref('')
const listed = ref<Row[]>([])
const reading = ref(false)
const needle = ref('')
const auditedOnly = ref(false)

const source = computed<Source | undefined>(() => loaded.value?.sources.find((s) => s.source === props.source))
const review = computed(() => loaded.value?.reviews.get(props.source))
const labels = computed<Label[]>(() => loaded.value?.labels.filter((label) => label.source === props.source) ?? [])
const labelled = computed(() => new Map(labels.value.map((label) => [label.row, label])))

const remaining = computed(() => (source.value ? source.value.rows - listed.value.length : 0))

const shown = computed(() => {
  const wanted = needle.value.trim().toLowerCase()
  if (auditedOnly.value) {
    return labels.value
      .filter((label) => !wanted || label.path.toLowerCase().includes(wanted) || label.summary.toLowerCase().includes(wanted))
      .map((label) => ({ row: label.row, path: label.path, label }))
  }
  return listed.value
    .filter((row) => !wanted || row.path.toLowerCase().includes(wanted))
    .map((row) => ({ ...row, label: labelled.value.get(row.row) }))
})

async function more(): Promise<void> {
  if (!source.value || reading.value || remaining.value <= 0) return
  reading.value = true
  try {
    const next = await readRows(source.value.offset + listed.value.length, Math.min(PAGE, remaining.value))
    listed.value = [...listed.value, ...next]
  } catch (error) {
    problem.value = String(error)
  } finally {
    reading.value = false
  }
}

async function start(): Promise<void> {
  listed.value = []
  problem.value = ''
  try {
    loaded.value = await corpus()
  } catch (error) {
    problem.value = String(error)
    return
  }
  if (!source.value) {
    problem.value = `No source called ${props.source}.`
    return
  }
  await more()
}

onMounted(start)
watch(() => props.source, start)
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <template v-if="source">
    <h2>{{ source.source }}</h2>
    <div class="tally-row">
      <div class="stat"><b>{{ count(source.rows) }}</b><span>tasks</span></div>
      <div class="stat"><b>{{ bytes(source.size) }}</b><span>parquet</span></div>
      <div class="stat"><b>{{ labels.length }}</b><span>audited</span></div>
      <div class="stat" v-if="review"><b><Verdict :value="review.shellsim_verdict" /></b><span>shellsim</span></div>
      <div class="stat" v-if="review && review.cheapest_unlock !== 'none'"><b><span class="chip">{{ review.cheapest_unlock }}</span></b><span>cheapest unlock</span></div>
    </div>

    <div class="panel review" v-if="review">
      <div>
        <h3 style="margin-top: 0">What it is</h3>
        <p>{{ review.template_description }}</p>
      </div>
      <div>
        <h3 style="margin-top: 0">Quality</h3>
        <p>{{ review.quality_notes }}</p>
        <h3 v-if="review.unlock_notes">What shellsim would need</h3>
        <p v-if="review.unlock_notes">{{ review.unlock_notes }}</p>
      </div>
    </div>

    <h3>Tasks</h3>
    <div class="controls">
      <input type="search" v-model="needle" placeholder="Filter by task name" />
      <label><input type="checkbox" v-model="auditedOnly" /> audited only</label>
      <span class="tally">
        <template v-if="auditedOnly">{{ shown.length }} of {{ labels.length }} audited</template>
        <template v-else>{{ shown.length }} of {{ count(listed.length) }} listed, {{ count(source.rows) }} in all</template>
      </span>
    </div>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Row</th>
            <th>Audit</th>
            <th v-if="auditedOnly">Interesting</th>
            <th v-if="auditedOnly">Hack risk</th>
            <th v-if="auditedOnly">Summary</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in shown" :key="item.row">
            <td><RouterLink :to="taskPath(item.row)">{{ item.path }}</RouterLink></td>
            <td class="num">{{ count(item.row) }}</td>
            <td>
              <template v-if="item.label">
                <Verdict :value="item.label.shellsim_now" prefix="shellsim: " />
                <span class="chip">{{ item.label.task_kind }}</span>
              </template>
            </td>
            <td v-if="auditedOnly && item.label"><Score :value="item.label.interesting" /></td>
            <td v-if="auditedOnly && item.label"><Score :value="item.label.hack_risk" :warn="item.label.hack_risk >= 4" /></td>
            <td v-if="auditedOnly && item.label">{{ item.label.summary }}</td>
          </tr>
        </tbody>
      </table>
      <p class="empty" v-if="shown.length === 0 && !reading">No task in what is listed matches. Load more, or search the whole dataset.</p>
    </div>
    <div class="more" v-if="!auditedOnly">
      <button class="button" type="button" @click="more" :disabled="reading || remaining <= 0">
        {{ reading ? 'Reading…' : remaining > 0 ? `Load ${Math.min(PAGE, remaining)} more` : 'Every task is listed' }}
      </button>
      <RouterLink :to="{ path: '/find', query: { q: needle } }" v-if="needle">Search the whole dataset for “{{ needle }}”</RouterLink>
    </div>
  </template>
  <p class="working" v-else-if="!problem">Reading the manifest…</p>
</template>
