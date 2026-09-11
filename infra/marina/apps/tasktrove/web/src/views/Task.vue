<script setup lang="ts">
// One task: its files, out of the tar the dataset row holds.
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { bytes, corpus, count, sourceOf, type Corpus } from '../corpus'
import { task as readTask, type Task } from '../trove'
import { sourcePath } from '../routes'
import LabelCard from '../components/LabelCard.vue'
import Prose from '@marina/Prose.vue'

const props = defineProps<{ row: number }>()
const route = useRoute()
const router = useRouter()

const loaded = ref<Corpus>()
const opened = ref<Task>()
const problem = ref('')
const wrap = ref(true)
const rendered = ref(true)

const source = computed(() => (loaded.value ? sourceOf(loaded.value.sources, props.row) : undefined))
const label = computed(() => loaded.value?.labelled.get(props.row))

const files = computed(() => opened.value?.files.filter((file) => !file.directory) ?? [])
const selectedPath = computed(() => (typeof route.query.file === 'string' ? route.query.file : undefined))
const selected = computed(() => files.value.find((file) => file.path === selectedPath.value) ?? files.value.find((file) => file.path === 'instruction.md') ?? files.value[0])
const markdown = computed(() => selected.value?.path.endsWith('.md') ?? false)

function pick(path: string): void {
  router.replace({ query: { ...route.query, file: path } })
}

async function start(): Promise<void> {
  opened.value = undefined
  problem.value = ''
  try {
    loaded.value = await corpus()
    opened.value = await readTask(props.row)
  } catch (error) {
    problem.value = String(error)
  }
}

onMounted(start)
watch(() => props.row, start)
</script>

<template>
  <p class="problem" v-if="problem">{{ problem }}</p>
  <p class="note" v-if="source">
    <RouterLink :to="sourcePath(source.source)">{{ source.source }}</RouterLink>
    · row {{ count(row) }}
  </p>
  <h2 v-if="opened">{{ opened.path }}</h2>
  <h2 v-else>Task {{ count(row) }}</h2>

  <LabelCard v-if="label" :label="label" />

  <div class="task" v-if="opened" style="margin-top: 1rem">
    <div>
      <div class="viewer heading"><span>{{ files.length }} files</span><span class="num">{{ bytes(opened.size) }} packed</span></div>
      <ul class="files">
        <li v-for="file in files" :key="file.path">
          <a href="#" @click.prevent="pick(file.path)" :aria-current="file === selected ? 'true' : undefined">
            <span>{{ file.path }}</span><span class="num">{{ bytes(file.size) }}</span>
          </a>
        </li>
      </ul>
    </div>
    <div class="viewer" v-if="selected">
      <div class="heading">
        <span>{{ selected.path }}</span>
        <span>
          <label v-if="markdown"><input type="checkbox" v-model="rendered" /> rendered</label>
          <label v-if="!markdown || !rendered"><input type="checkbox" v-model="wrap" /> wrap</label>
        </span>
      </div>
      <div class="panel" v-if="markdown && rendered && selected.text !== undefined"><Prose :text="selected.text" /></div>
      <pre v-else-if="selected.text !== undefined" :data-wrap="wrap ? 'true' : 'false'">{{ selected.text }}</pre>
      <p class="empty" v-else>Not text: {{ bytes(selected.size) }} of binary.</p>
    </div>
  </div>
  <p class="working" v-else-if="!problem">Fetching the task from Hugging Face…</p>
</template>
