<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Flat } from '../composables/useRun'

const props = defineProps<{ summary: Flat; config: Flat }>()

const search = ref('')

function filtered(obj: Flat): [string, unknown][] {
  const q = search.value.toLowerCase()
  return Object.entries(obj)
    .filter(([k]) => k.toLowerCase().includes(q))
    .sort(([a], [b]) => a.localeCompare(b))
}

const summaryRows = computed(() => filtered(props.summary))
const configRows = computed(() => filtered(props.config))
</script>

<template>
  <input
    v-model="search"
    placeholder="filter keys…"
    class="mb-4 block w-full max-w-lg rounded border border-surface-border bg-surface-raised px-3 py-1.5 text-sm"
  />
  <div class="grid grid-cols-1 gap-8 md:grid-cols-2">
    <section>
      <h3 class="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">summary metrics</h3>
      <table class="w-full border-collapse text-xs">
        <tbody>
          <tr v-for="[k, v] in summaryRows" :key="k" class="border-b border-surface-border/60 align-top">
            <td class="w-[46%] py-1 pr-2 font-mono text-text-secondary">{{ k }}</td>
            <td class="break-all py-1 font-mono">{{ v }}</td>
          </tr>
        </tbody>
      </table>
    </section>
    <section>
      <h3 class="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">config</h3>
      <table class="w-full border-collapse text-xs">
        <tbody>
          <tr v-for="[k, v] in configRows" :key="k" class="border-b border-surface-border/60 align-top">
            <td class="w-[46%] py-1 pr-2 font-mono text-text-secondary">{{ k }}</td>
            <td class="break-all py-1 font-mono">{{ v }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>
