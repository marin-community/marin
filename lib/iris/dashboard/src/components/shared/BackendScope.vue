<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useBackends } from '@/composables/useBackends'

// Above this threshold we switch from a plain <select> to a searchable combo.
const COMBOBOX_THRESHOLD = 8

const route = useRoute()
const router = useRouter()
const { backends } = useBackends()

const selectedBackend = computed(() => {
  const id = route.query.backend
  return Array.isArray(id) ? (id[0] ?? '') : (id ?? '')
})

const searchTerm = ref('')

const filteredBackends = computed(() => {
  if (!searchTerm.value) return backends.value
  const lower = searchTerm.value.toLowerCase()
  return backends.value.filter(
    b => b.id.toLowerCase().includes(lower) || b.name.toLowerCase().includes(lower),
  )
})

const isCombobox = computed(() => backends.value.length > COMBOBOX_THRESHOLD)

function selectBackend(id: string) {
  searchTerm.value = ''
  router.replace({
    query: {
      ...route.query,
      backend: id || undefined,
    },
  })
}

function handleSelectChange(event: Event) {
  selectBackend((event.target as HTMLSelectElement).value)
}
</script>

<template>
  <template v-if="backends.length > 1">
    <!-- Simple <select> for small backend counts -->
    <select
      v-if="!isCombobox"
      :value="selectedBackend"
      aria-label="Scope to backend"
      class="px-2 py-1 text-sm border border-surface-border rounded bg-surface text-text
             focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      @change="handleSelectChange"
    >
      <option value="">All backends</option>
      <option v-for="b in backends" :key="b.id" :value="b.id">
        {{ b.name || b.id }}
      </option>
    </select>

    <!-- Searchable combobox for large backend counts -->
    <div v-else class="relative">
      <input
        v-model="searchTerm"
        type="text"
        :placeholder="selectedBackend ? (backends.find(b => b.id === selectedBackend)?.name ?? selectedBackend) : 'All backends'"
        aria-label="Scope to backend"
        class="w-44 px-2 py-1 text-sm border border-surface-border rounded bg-surface text-text
               placeholder:text-text-secondary
               focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      />
      <div
        v-if="searchTerm"
        class="absolute z-50 mt-1 w-full rounded border border-surface-border bg-surface shadow-lg"
      >
        <button
          class="w-full px-3 py-1.5 text-left text-sm hover:bg-surface-raised"
          @click="selectBackend('')"
        >
          All backends
        </button>
        <button
          v-for="b in filteredBackends"
          :key="b.id"
          class="w-full px-3 py-1.5 text-left text-sm hover:bg-surface-raised"
          :class="b.id === selectedBackend ? 'text-accent font-medium' : 'text-text'"
          @click="selectBackend(b.id)"
        >
          {{ b.name || b.id }}
        </button>
        <div
          v-if="filteredBackends.length === 0"
          class="px-3 py-1.5 text-sm text-text-muted"
        >
          No backends match
        </div>
      </div>
    </div>
  </template>
</template>
