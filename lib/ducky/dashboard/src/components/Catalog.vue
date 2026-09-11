<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

interface CatalogView {
  schema: string
  name: string
  qualified_name: string
  description: string
  insert_sql: string
}
interface CatalogExample {
  title: string
  description: string
  sql: string
}

const emit = defineEmits<{ select: [string] }>()

const views = ref<CatalogView[]>([])
const examples = ref<CatalogExample[]>([])
const open = ref(true)
// Examples are collapsed by default (they're a long list); expand to browse them.
const examplesOpen = ref(false)

// Group views by their schema (finelog, datakit, …) so the panel reads as one block per
// data source rather than a flat list.
const groups = computed(() => {
  const bySchema = new Map<string, CatalogView[]>()
  for (const view of views.value) {
    const list = bySchema.get(view.schema) ?? []
    list.push(view)
    bySchema.set(view.schema, list)
  }
  return [...bySchema.entries()].map(([schema, items]) => ({ schema, items }))
})

const hasCatalog = computed(() => views.value.length > 0 || examples.value.length > 0)

onMounted(async () => {
  try {
    const cfg = await (await fetch('api/catalog')).json()
    views.value = cfg.views ?? []
    examples.value = cfg.examples ?? []
  } catch (e) {
    /* catalog is best-effort; the panel just stays empty */
  }
})
</script>

<template>
  <section v-if="hasCatalog" class="rounded-lg border border-surface-border">
    <button
      class="flex w-full items-center justify-between px-3 py-2 text-sm font-medium text-text-secondary hover:bg-surface-raised"
      @click="open = !open"
    >
      <span>Pre-baked sources &amp; queries</span>
      <span class="text-text-muted">{{ open ? '▾' : '▸' }}</span>
    </button>

    <div v-if="open" class="flex flex-col gap-3 border-t border-surface-border px-3 py-3">
      <div v-for="group in groups" :key="group.schema" class="flex flex-col gap-1.5">
        <h3 class="text-xs font-semibold uppercase tracking-wide text-text-muted">{{ group.schema }}</h3>
        <div class="flex flex-wrap gap-1.5">
          <button
            v-for="view in group.items"
            :key="view.qualified_name"
            class="rounded-md border border-surface-border bg-surface-raised px-2 py-1 font-mono text-xs text-text hover:border-accent hover:text-accent"
            :title="view.description + '\nClick to query: ' + view.insert_sql"
            @click="emit('select', view.insert_sql)"
          >
            {{ view.qualified_name }}
          </button>
        </div>
      </div>

      <div v-if="examples.length" class="flex flex-col gap-1.5">
        <button
          class="flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-text-muted hover:text-text-secondary"
          @click="examplesOpen = !examplesOpen"
        >
          <span>{{ examplesOpen ? '▾' : '▸' }}</span>
          <span>Example queries</span>
        </button>
        <div v-if="examplesOpen" class="flex flex-wrap gap-1.5">
          <button
            v-for="example in examples"
            :key="example.title"
            class="rounded-md border border-surface-border bg-surface-raised px-2 py-1 text-xs text-text-secondary hover:border-accent hover:text-accent"
            :title="example.description"
            @click="emit('select', example.sql)"
          >
            {{ example.title }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
