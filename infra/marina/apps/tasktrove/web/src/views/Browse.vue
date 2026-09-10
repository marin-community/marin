<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { corpus, count, type CatalogTask, type Corpus } from '../corpus'
import { taskPath } from '../routes'

const PAGE = 60
const route = useRoute()
const router = useRouter()
const loaded = ref<Corpus>()
const problem = ref('')
const needle = ref(typeof route.query.q === 'string' ? route.query.q : '')
const mode = ref(typeof route.query.mode === 'string' ? route.query.mode : '')
const environment = ref(typeof route.query.env === 'string' ? route.query.env : '')
const language = ref(typeof route.query.lang === 'string' ? route.query.lang : '')
const visible = ref(PAGE)

onMounted(async () => {
  try {
    loaded.value = await corpus()
  } catch (error) {
    problem.value = String(error)
  }
})

function distinct(pick: (task: CatalogTask) => string): string[] {
  return [...new Set((loaded.value?.tasks ?? []).map(pick).filter(Boolean))].sort()
}

const modes = computed(() => distinct((task) => task.mode))
const environments = computed(() => distinct((task) => task.environment))
const languages = computed(() => distinct((task) => task.language))
const filtered = computed(() => {
  const query = needle.value.trim().toLowerCase()
  return (loaded.value?.tasks ?? []).filter((task) => {
    if (mode.value && task.mode !== mode.value) return false
    if (environment.value && task.environment !== environment.value) return false
    if (language.value && task.language !== language.value) return false
    if (query && !`${task.path} ${task.source} ${task.converter} ${task.tags.join(' ')}`.toLowerCase().includes(query)) {
      return false
    }
    return true
  })
})
const shown = computed(() => filtered.value.slice(0, visible.value))

watch([needle, mode, environment, language], () => {
  visible.value = PAGE
  router.replace({
    query: {
      ...(needle.value.trim() ? { q: needle.value.trim() } : {}),
      ...(mode.value ? { mode: mode.value } : {}),
      ...(environment.value ? { env: environment.value } : {}),
      ...(language.value ? { lang: language.value } : {}),
    },
  })
})
</script>

<template>
  <header class="browse-heading">
    <div>
      <p class="eyebrow">Normalized examples</p>
      <h1>Browse tasks</h1>
      <p>
        Scroll the rows of a generated clean Parquet sample covering every verifier-mode and Docker-environment pair.
        Open any row to inspect the normalized task archive stored in its <code>task_binary</code> column.
      </p>
    </div>
    <div class="coverage" v-if="loaded">
      <b>{{ count(loaded.tasks.length) }}</b><span>sampled tasks</span>
      <b>{{ modes.length }}</b><span>modes</span>
      <b>{{ environments.length }}</b><span>environments</span>
    </div>
  </header>

  <p class="problem" v-if="problem">{{ problem }}</p>
  <template v-if="loaded">
    <div class="filters">
      <input type="search" v-model="needle" placeholder="Search source, task, converter, or tag" />
      <select v-model="mode" aria-label="Verifier mode">
        <option value="">Every verifier mode</option>
        <option v-for="value in modes" :key="value" :value="value">{{ value }}</option>
      </select>
      <select v-model="environment" aria-label="Docker environment">
        <option value="">Every Docker environment</option>
        <option v-for="value in environments" :key="value" :value="value">{{ value }}</option>
      </select>
      <select v-model="language" aria-label="Language">
        <option value="">Every language</option>
        <option v-for="value in languages" :key="value" :value="value">{{ value }}</option>
      </select>
      <span>{{ count(filtered.length) }} matches</span>
    </div>

    <div class="task-grid">
      <RouterLink v-for="task in shown" :key="task.row" :to="taskPath(task.row)" class="task-card">
        <div class="task-card-top">
          <span class="mode">{{ task.mode }}</span>
          <span class="language" v-if="task.language">{{ task.language }}</span>
        </div>
        <h2>{{ task.path }}</h2>
        <p>{{ task.source }}</p>
        <div class="task-meta">
          <span>{{ task.environment }}</span>
          <span>{{ task.converter }}</span>
        </div>
        <div class="tags"><span v-for="tag in task.tags.slice(0, 5)" :key="tag">{{ tag }}</span></div>
      </RouterLink>
    </div>
    <p class="empty" v-if="shown.length === 0">No sampled task matches these filters.</p>
    <div class="more" v-if="visible < filtered.length">
      <button type="button" @click="visible += PAGE">Show {{ Math.min(PAGE, filtered.length - visible) }} more</button>
    </div>
  </template>
  <p class="working" v-else-if="!problem">Reading task samples…</p>
</template>
