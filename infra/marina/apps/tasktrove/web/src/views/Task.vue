<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import Prose from '@marina/Prose.vue'
import { archive, count, task as parquetTask, type ParquetTask } from '../corpus'
import { entries, gunzip, text, type Entry } from '../tar'

const props = defineProps<{ id: string }>()
const route = useRoute()
const router = useRouter()
const task = ref<ParquetTask>()
const files = ref<Entry[]>([])
const problem = ref('')
const wrap = ref(true)
const rendered = ref(true)

const selectedPath = computed(() => (typeof route.query.file === 'string' ? route.query.file : undefined))
const selected = computed(
  () => files.value.find((file) => file.path === selectedPath.value) ?? files.value.find((file) => file.path === 'instruction.md') ?? files.value[0],
)
const selectedText = computed(() => (selected.value?.directory ? '' : text(selected.value?.bytes ?? new Uint8Array())))
const markdown = computed(() => selected.value?.path.endsWith('.md') ?? false)

function bytes(value: number): string {
  if (value < 1_000) return `${value} B`
  if (value < 1_000_000) return `${(value / 1_000).toFixed(1)} KB`
  return `${(value / 1_000_000).toFixed(1)} MB`
}

function pick(path: string): void {
  router.replace({ query: { file: path } })
}

async function start(): Promise<void> {
  problem.value = ''
  files.value = []
  try {
    const row = Number(props.id)
    if (!Number.isSafeInteger(row) || row < 0) throw new Error(`Invalid Parquet row ${props.id}.`)
    task.value = await parquetTask(row)
    const packed = await archive(row)
    files.value = entries(await gunzip(packed)).sort((a, b) => a.path.localeCompare(b.path))
  } catch (error) {
    problem.value = String(error)
  }
}

onMounted(start)
watch(() => props.id, start)
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <template v-if="task">
    <RouterLink class="back" to="/browse">← Browse tasks</RouterLink>
    <header class="task-heading">
      <div>
        <p class="eyebrow">{{ task.mode }} · {{ task.environment }}</p>
        <h1>{{ task.path }}</h1>
        <p>{{ task.source }}</p>
      </div>
      <dl>
        <dt>Converter</dt><dd>{{ task.converter }}</dd>
        <dt>Language</dt><dd>{{ task.language || 'not specified' }}</dd>
        <dt>Dockerfile</dt><dd><code>{{ task.dockerfile_id.slice(0, 12) }}</code></dd>
        <dt>Parquet row</dt><dd>{{ count(task.row) }}</dd>
        <dt>Oracle</dt><dd>{{ task.has_solution ? 'included separately' : 'none' }}</dd>
      </dl>
    </header>
    <div class="tags"><span v-for="tag in task.tags" :key="tag">{{ tag }}</span></div>

    <div class="task-viewer">
      <aside>
        <div class="file-count">{{ count(files.filter((file) => !file.directory).length) }} files</div>
        <button
          v-for="file in files.filter((candidate) => !candidate.directory)"
          :key="file.path"
          type="button"
          :aria-current="file === selected ? 'true' : undefined"
          @click="pick(file.path)"
        >
          <span>{{ file.path }}</span><small>{{ bytes(file.size) }}</small>
        </button>
      </aside>
      <section class="file-view" v-if="selected">
        <div class="file-toolbar">
          <code>{{ selected.path }}</code>
          <span>
            <label v-if="markdown"><input type="checkbox" v-model="rendered" /> rendered</label>
            <label v-if="!markdown || !rendered"><input type="checkbox" v-model="wrap" /> wrap</label>
          </span>
        </div>
        <div class="rendered" v-if="markdown && rendered && selectedText !== undefined"><Prose :text="selectedText" /></div>
        <pre v-else-if="selectedText !== undefined" :data-wrap="wrap ? 'true' : 'false'">{{ selectedText }}</pre>
        <p class="empty" v-else>Binary file, {{ bytes(selected.size) }}</p>
      </section>
    </div>
  </template>
  <p class="working" v-else-if="!problem">Reading task_binary from Parquet…</p>
</template>
