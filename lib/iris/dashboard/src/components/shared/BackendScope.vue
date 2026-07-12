<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useBackends, firstQueryValue } from '@/composables/useBackends'

// Above this threshold we switch from a plain <select> to a searchable combo.
const COMBOBOX_THRESHOLD = 8

// An option value encodes the target kind so a selection writes the matching
// query param. Kept in one place so the encode and decode sites stay in sync.
const BACKEND_SCOPE_PREFIX = 'backend:'
const CLUSTER_SCOPE_PREFIX = 'cluster:'

const route = useRoute()
const router = useRouter()
const { backends, peers, ensurePeers } = useBackends()

// Load the peer roster so a 1-backend + N-peer deployment still offers the
// selector; inert (empty) on a single-cluster deployment.
onMounted(ensurePeers)

// One combined scope list over execution targets: local backends (scoped via
// ?backend=) and federation peers (scoped via ?cluster=). The option value
// carries the kind so a selection writes the right query param.
interface ScopeOption {
  value: string
  label: string
}

const options = computed<ScopeOption[]>(() => [
  ...backends.value.map(b => ({ value: `${BACKEND_SCOPE_PREFIX}${b.id}`, label: b.name || b.id })),
  ...peers.value.map(p => ({ value: `${CLUSTER_SCOPE_PREFIX}${p.peerId}`, label: `${p.peerId} (peer)` })),
])

const targetCount = computed(() => backends.value.length + peers.value.length)
const isCombobox = computed(() => targetCount.value > COMBOBOX_THRESHOLD)

const selectedValue = computed(() => {
  const backend = firstQueryValue(route.query.backend)
  if (backend) return `${BACKEND_SCOPE_PREFIX}${backend}`
  const cluster = firstQueryValue(route.query.cluster)
  if (cluster) return `${CLUSTER_SCOPE_PREFIX}${cluster}`
  return ''
})

const selectedLabel = computed(
  () => options.value.find(o => o.value === selectedValue.value)?.label ?? 'All targets',
)

const searchTerm = ref('')

const filteredOptions = computed(() => {
  if (!searchTerm.value) return options.value
  const lower = searchTerm.value.toLowerCase()
  return options.value.filter(o => o.label.toLowerCase().includes(lower))
})

/** Apply a scope selection. `value` is '' (all), 'backend:<id>', or 'cluster:<id>'. */
function applyScope(value: string) {
  searchTerm.value = ''
  // A target is either a backend or a peer — never both — so set one and clear
  // the other (undefined drops the param from the URL).
  const backend = value.startsWith(BACKEND_SCOPE_PREFIX) ? value.slice(BACKEND_SCOPE_PREFIX.length) : undefined
  const cluster = value.startsWith(CLUSTER_SCOPE_PREFIX) ? value.slice(CLUSTER_SCOPE_PREFIX.length) : undefined
  router.replace({ query: { ...route.query, backend, cluster } })
}

function handleSelectChange(event: Event) {
  applyScope((event.target as HTMLSelectElement).value)
}
</script>

<template>
  <template v-if="targetCount > 1">
    <!-- Simple <select> for small target counts -->
    <select
      v-if="!isCombobox"
      :value="selectedValue"
      aria-label="Scope to backend or peer"
      class="px-2 py-1 text-sm border border-surface-border rounded bg-surface text-text
             focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent"
      @change="handleSelectChange"
    >
      <option value="">All targets</option>
      <option v-for="o in options" :key="o.value" :value="o.value">
        {{ o.label }}
      </option>
    </select>

    <!-- Searchable combobox for large target counts -->
    <div v-else class="relative">
      <input
        v-model="searchTerm"
        type="text"
        :placeholder="selectedLabel"
        aria-label="Scope to backend or peer"
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
          @click="applyScope('')"
        >
          All targets
        </button>
        <button
          v-for="o in filteredOptions"
          :key="o.value"
          class="w-full px-3 py-1.5 text-left text-sm hover:bg-surface-raised"
          :class="o.value === selectedValue ? 'text-accent font-medium' : 'text-text'"
          @click="applyScope(o.value)"
        >
          {{ o.label }}
        </button>
        <div
          v-if="filteredOptions.length === 0"
          class="px-3 py-1.5 text-sm text-text-muted"
        >
          No targets match
        </div>
      </div>
    </div>
  </template>
</template>
