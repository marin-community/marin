<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { count, manifest, rowCount, tasks, type Manifest, type ParquetTask } from '../corpus'
import { taskPath } from '../routes'

const PAGE = 60
const dataset = ref<Manifest>()
const rows = ref<ParquetTask[]>([])
const total = ref(0)
const problem = ref('')
const working = ref(false)

const modes = computed(() => Object.keys(dataset.value?.by_mode ?? {}).length)
const environments = computed(() => Object.keys(dataset.value?.dockerfiles ?? {}).length)

async function loadMore(): Promise<void> {
  if (working.value || rows.value.length >= total.value) return
  working.value = true
  try {
    const start = rows.value.length
    rows.value.push(...(await tasks(start, Math.min(start + PAGE, total.value))))
  } catch (error) {
    problem.value = String(error)
  } finally {
    working.value = false
  }
}

onMounted(async () => {
  try {
    const [loadedManifest, loadedRows] = await Promise.all([manifest(), rowCount()])
    dataset.value = loadedManifest
    total.value = loadedRows
    await loadMore()
  } catch (error) {
    problem.value = String(error)
  }
})
</script>

<template>
  <header class="browse-heading">
    <div>
      <p class="eyebrow">Final output</p>
      <h1>Parquet viewer</h1>
      <p>
        Rows come directly from the final clean Parquet file. Open any row to inspect the normalized task archive stored
        in its <code>task_binary</code> column.
      </p>
    </div>
    <div class="coverage" v-if="dataset">
      <b>{{ count(total) }}</b><span>Parquet rows</span>
      <b>{{ modes }}</b><span>modes</span>
      <b>{{ environments }}</b><span>environments</span>
    </div>
  </header>

  <p class="problem" v-if="problem">{{ problem }}</p>
  <div class="parquet-position" v-if="rows.length">
    Showing rows 1–{{ count(rows.length) }} of {{ count(total) }}
  </div>
  <div class="task-grid">
    <RouterLink v-for="item in rows" :key="item.row" :to="taskPath(item.row)" class="task-card">
      <div class="task-card-top">
        <span class="mode">{{ item.mode }}</span>
        <span class="language" v-if="item.language">{{ item.language }}</span>
      </div>
      <h2>{{ item.path }}</h2>
      <p>{{ item.source }}</p>
      <div class="task-meta">
        <span>{{ item.environment }}</span>
        <span>{{ item.converter }}</span>
      </div>
      <div class="tags"><span v-for="tag in item.tags.slice(0, 5)" :key="tag">{{ tag }}</span></div>
    </RouterLink>
  </div>
  <div class="more" v-if="rows.length < total">
    <button type="button" @click="loadMore" :disabled="working">
      {{ working ? 'Reading Parquet…' : `Show next ${Math.min(PAGE, total - rows.length)} rows` }}
    </button>
  </div>
  <p class="working" v-else-if="working">Reading the Parquet footer…</p>
</template>
