<script setup lang="ts">
// The audited sample, filtered by what the audit said.
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { corpus, type Corpus } from '../corpus'
import { sourcePath, taskPath } from '../routes'
import Verdict from '../components/Verdict.vue'
import Score from '../components/Score.vue'

const loaded = ref<Corpus>()
const problem = ref('')
const needle = ref('')
const kind = ref('')
const now = ref('')
const mechanism = ref('')
const need = ref('')
const hack = ref(0)

onMounted(async () => {
  try {
    loaded.value = await corpus()
  } catch (error) {
    problem.value = String(error)
  }
})

function distinct(pick: (label: Corpus['labels'][number]) => string[]): string[] {
  if (!loaded.value) return []
  const seen = new Set<string>()
  for (const label of loaded.value.labels) for (const value of pick(label)) seen.add(value)
  return [...seen].sort()
}

const kinds = computed(() => distinct((label) => [label.task_kind]))
const mechanisms = computed(() => distinct((label) => [label.verifier_mechanism]))
const needs = computed(() => distinct((label) => label.agent_needs))

const rows = computed(() => {
  if (!loaded.value) return []
  const wanted = needle.value.trim().toLowerCase()
  return loaded.value.labels.filter((label) => {
    if (kind.value && label.task_kind !== kind.value) return false
    if (now.value && label.shellsim_now !== now.value) return false
    if (mechanism.value && label.verifier_mechanism !== mechanism.value) return false
    if (need.value && !label.agent_needs.includes(need.value)) return false
    if (label.hack_risk < hack.value) return false
    if (wanted && !`${label.path} ${label.source} ${label.summary}`.toLowerCase().includes(wanted)) return false
    return true
  })
})
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <p class="note">
    The tasks the audit read: eleven or so from every source, each classified
    by hand. Filter by what the task is, what it needs to run, and how it is
    verified.
  </p>
  <div class="controls">
    <input type="search" v-model="needle" placeholder="Filter by name, source or summary" />
    <select v-model="kind" aria-label="task kind"><option value="">any kind</option><option v-for="k in kinds" :key="k" :value="k">{{ k }}</option></select>
    <select v-model="now" aria-label="shellsim now"><option value="">any shellsim verdict</option><option value="yes">runs today</option><option value="partial">partial</option><option value="no">does not run</option></select>
    <select v-model="mechanism" aria-label="verifier"><option value="">any verifier</option><option v-for="m in mechanisms" :key="m" :value="m">{{ m }}</option></select>
    <select v-model="need" aria-label="agent needs"><option value="">any need</option><option v-for="n in needs" :key="n" :value="n">{{ n }}</option></select>
    <select v-model.number="hack" aria-label="hack risk"><option :value="0">any hack risk</option><option :value="3">hack risk ≥ 3</option><option :value="4">hack risk ≥ 4</option><option :value="5">hack risk 5</option></select>
    <span class="tally" v-if="loaded">{{ rows.length }} of {{ loaded.labels.length }}</span>
  </div>
  <div class="scroll" v-if="loaded">
    <table>
      <thead>
        <tr><th>Task</th><th>Source</th><th>Kind</th><th>shellsim</th><th>Interesting</th><th>Defined</th><th>Hack</th><th>Summary</th></tr>
      </thead>
      <tbody>
        <tr v-for="label in rows" :key="label.id">
          <td><RouterLink :to="taskPath(label.row)">{{ label.path }}</RouterLink></td>
          <td><RouterLink :to="sourcePath(label.source)">{{ label.source }}</RouterLink></td>
          <td><span class="chip">{{ label.task_kind }}</span></td>
          <td><Verdict :value="label.shellsim_now" /></td>
          <td><Score :value="label.interesting" /></td>
          <td><Score :value="label.well_defined" /></td>
          <td><Score :value="label.hack_risk" :warn="label.hack_risk >= 4" /></td>
          <td>{{ label.summary }}</td>
        </tr>
      </tbody>
    </table>
    <p class="empty" v-if="rows.length === 0">No audited task matches that filter.</p>
  </div>
</template>
