<script setup lang="ts">
import { onMounted } from 'vue'
import { useRuns } from '../composables/useRuns'

const emit = defineEmits<{ select: [name: string] }>()
const { entity, project, user, search, entities, projects, users, runs, loading, loadEntities, fetchRuns } = useRuns()

onMounted(async () => {
  await loadEntities()
  fetchRuns()
})

const GLYPH: Record<string, string> = {
  running: '🟢',
  finished: '🏁',
  crashed: '💥',
  failed: '❌',
  killed: '🔪',
  preempted: '⏸️',
}
const glyph = (state: string) => GLYPH[state] ?? '·'
</script>

<template>
  <aside class="flex w-[340px] shrink-0 flex-col gap-2 bg-accent p-4 text-white">
    <div class="text-2xl font-bold tracking-wide">buoy</div>

    <label class="mt-1 text-xs uppercase tracking-wide text-white/70">entity</label>
    <input v-model="entity" list="ents" class="rounded px-2 py-1 text-text" @change="fetchRuns" />
    <datalist id="ents"><option v-for="e in entities" :key="e" :value="e" /></datalist>

    <label class="text-xs uppercase tracking-wide text-white/70">project</label>
    <input v-model="project" list="projs" class="rounded px-2 py-1 text-text" />
    <datalist id="projs"><option v-for="p in projects" :key="p" :value="p" /></datalist>

    <label class="text-xs uppercase tracking-wide text-white/70">user</label>
    <input v-model="user" list="usrs" class="rounded px-2 py-1 text-text" @change="fetchRuns" />
    <datalist id="usrs"><option v-for="u in users" :key="u" :value="u" /></datalist>

    <input
      v-model="search"
      placeholder="search runs…"
      class="mt-1 rounded px-2 py-1 text-text"
      @input="fetchRuns"
    />

    <div class="mt-1 text-xs uppercase tracking-wide text-white/70">
      {{ loading ? 'loading…' : `${runs.length} runs` }}
    </div>

    <div class="-mx-1 flex-1 overflow-auto px-1">
      <button
        v-for="run in runs"
        :key="run.id"
        class="w-full break-all rounded px-2 py-1 text-left font-mono text-xs hover:bg-white/10"
        @click="emit('select', run.name)"
      >
        <span>{{ glyph(run.state) }}</span> {{ run.name }}
      </button>
    </div>
  </aside>
</template>
