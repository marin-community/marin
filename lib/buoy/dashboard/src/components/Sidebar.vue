<script setup lang="ts">
import { onMounted } from 'vue'
import { useRuns } from '../composables/useRuns'

const props = defineProps<{ initialEntity?: string; initialProject?: string; initialUser?: string }>()
const emit = defineEmits<{ select: [entity: string, project: string, name: string]; collapse: [] }>()
const { entity, project, user, search, entities, projects, users, runs, loading, loadEntities, fetchRuns } = useRuns()

onMounted(async () => {
  await loadEntities()
  // A deep-link prefills the picker; the entity/project watches load the rest.
  if (props.initialEntity) entity.value = props.initialEntity
  if (props.initialProject) project.value = props.initialProject
  if (props.initialUser) user.value = props.initialUser
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
    <div class="flex items-center justify-between">
      <div class="text-2xl font-bold tracking-wide">buoy</div>
      <button
        class="rounded bg-white/15 px-2 py-0.5 text-sm hover:bg-white/30"
        title="hide sidebar"
        @click="emit('collapse')"
      >
        «
      </button>
    </div>

    <label class="mt-1 text-xs uppercase tracking-wide text-white/70">entity</label>
    <input v-model="entity" list="ents" class="rounded bg-surface-raised px-2 py-1 text-text" @change="fetchRuns" />
    <datalist id="ents"><option v-for="e in entities" :key="e" :value="e" /></datalist>

    <label class="text-xs uppercase tracking-wide text-white/70">project</label>
    <input v-model="project" list="projs" class="rounded bg-surface-raised px-2 py-1 text-text" />
    <datalist id="projs"><option v-for="p in projects" :key="p" :value="p" /></datalist>

    <label class="text-xs uppercase tracking-wide text-white/70">user</label>
    <input v-model="user" list="usrs" class="rounded bg-surface-raised px-2 py-1 text-text" @change="fetchRuns" />
    <datalist id="usrs"><option v-for="u in users" :key="u" :value="u" /></datalist>

    <input
      v-model="search"
      placeholder="search runs…"
      class="mt-1 rounded bg-surface-raised px-2 py-1 text-text"
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
        @click="emit('select', entity, project, run.name)"
      >
        <span>{{ glyph(run.state) }}</span> {{ run.name }}
      </button>
    </div>
  </aside>
</template>
