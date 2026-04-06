import { createApp, ref, computed, watch, onMounted } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { humanBytes, humanCost, humanCount } from './format.js'
import { fetchOverview, fetchSavings, fetchRules, fetchSimulate } from './api.js'

// ---------------------------------------------------------------------------
// StorageClassBreakdown
// ---------------------------------------------------------------------------

const CLASS_COLORS = {
  STANDARD: 'bg-emerald-500',
  NEARLINE: 'bg-yellow-500',
  COLDLINE: 'bg-blue-500',
  ARCHIVE: 'bg-gray-500',
}

const CLASS_DOT_COLORS = {
  STANDARD: 'bg-emerald-400',
  NEARLINE: 'bg-yellow-400',
  COLDLINE: 'bg-blue-400',
  ARCHIVE: 'bg-gray-400',
}

const StorageClassBreakdown = {
  props: { breakdown: Array },
  setup(props) {
    const totalCost = computed(() =>
      props.breakdown.reduce((sum, b) => sum + b.monthly_cost_usd, 0)
    )
    const segments = computed(() =>
      props.breakdown
        .filter(b => b.monthly_cost_usd > 0)
        .map(b => ({
          ...b,
          pct: totalCost.value > 0 ? (b.monthly_cost_usd / totalCost.value) * 100 : 0,
          color: CLASS_COLORS[b.class] ?? 'bg-gray-500',
          dotColor: CLASS_DOT_COLORS[b.class] ?? 'bg-gray-400',
        }))
    )
    return { segments, humanBytes, humanCost }
  },
  template: `
    <div v-if="breakdown.length > 0" class="mt-3">
      <div class="flex h-2.5 rounded-full overflow-hidden bg-surface-sunken">
        <div v-for="seg in segments" :key="seg.class"
             :class="[seg.color, 'transition-all']"
             :style="{ width: seg.pct + '%' }" />
      </div>
      <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        <div v-for="seg in segments" :key="seg.class"
             class="flex items-center gap-1.5 text-xs text-text-secondary">
          <span :class="[seg.dotColor, 'w-2 h-2 rounded-full inline-block']" />
          <span>{{ seg.class }}</span>
          <span class="font-mono text-text-muted">{{ humanCost(seg.monthly_cost_usd) }}</span>
          <span class="text-text-muted">({{ humanBytes(seg.bytes) }})</span>
        </div>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// RegionCostCard
// ---------------------------------------------------------------------------

const RegionCostCard = {
  props: { region: Object, discount: Number },
  components: { StorageClassBreakdown },
  setup() {
    return { humanBytes, humanCost }
  },
  template: `
    <div class="rounded-lg border border-surface-border bg-surface-raised p-4">
      <div class="flex items-start justify-between">
        <div>
          <h3 class="text-sm font-semibold text-text">{{ region.region }}</h3>
          <p class="text-xs text-text-muted mt-0.5">{{ region.bucket }}</p>
        </div>
        <span class="text-xs text-text-muted px-2 py-0.5 rounded bg-surface-sunken">
          {{ region.continent }}
        </span>
      </div>
      <div class="mt-3 grid grid-cols-2 gap-3">
        <div>
          <div class="text-xs text-text-secondary">Size</div>
          <div class="text-sm font-mono font-semibold text-text">{{ humanBytes(region.total_bytes) }}</div>
        </div>
        <div>
          <div class="text-xs text-text-secondary">Monthly cost</div>
          <div class="text-sm font-mono font-semibold text-accent">{{ humanCost(region.total_monthly_cost_usd) }}</div>
          <div v-if="discount > 0" class="text-[10px] text-text-muted mt-0.5">
            after {{ Math.round(discount * 100) }}% discount
          </div>
        </div>
      </div>
      <StorageClassBreakdown :breakdown="region.by_storage_class" />
    </div>
  `,
}

// ---------------------------------------------------------------------------
// CostCalculator
// ---------------------------------------------------------------------------

const CostCalculator = {
  props: { excluded: Set, rules: Array },
  setup(props) {
    const loading = ref(false)
    const result = ref(null)
    const error = ref(null)
    let debounceTimer = null

    watch(
      () => [...props.excluded],
      (ids) => {
        if (debounceTimer) clearTimeout(debounceTimer)
        debounceTimer = setTimeout(async () => {
          if (ids.length === 0) { result.value = null; return }
          loading.value = true
          error.value = null
          try {
            result.value = await fetchSimulate(ids)
          } catch (e) {
            error.value = e.message || String(e)
          } finally {
            loading.value = false
          }
        }, 300)
      },
      { deep: true }
    )

    return { loading, result, error, humanBytes, humanCost, humanCount }
  },
  template: `
    <div class="rounded-lg border border-surface-border bg-surface-raised p-4 sticky top-6">
      <h2 class="text-sm font-semibold text-text uppercase tracking-wider mb-4">Cost Calculator</h2>
      <div class="space-y-4">
        <div class="text-sm text-text-secondary">
          <span class="font-mono text-text font-semibold">{{ excluded.size }}</span>
          rules selected for removal
        </div>

        <div v-if="loading" class="flex items-center gap-2 text-sm text-text-muted py-4">
          <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Calculating...
        </div>
        <div v-else-if="error" class="text-sm text-status-danger">{{ error }}</div>
        <template v-else-if="result">
          <div class="border-t border-surface-border pt-3 space-y-3">
            <div>
              <div class="text-xs text-text-secondary">Deletable objects</div>
              <div class="text-sm font-mono font-semibold text-text">{{ humanCount(result.totals.deletable_objects) }}</div>
            </div>
            <div>
              <div class="text-xs text-text-secondary">Deletable size</div>
              <div class="text-sm font-mono font-semibold text-text">{{ humanBytes(result.totals.deletable_bytes) }}</div>
            </div>
            <div>
              <div class="text-xs text-text-secondary">Monthly savings</div>
              <div class="text-lg font-mono font-bold text-status-success">{{ humanCost(result.totals.monthly_savings_usd) }}</div>
            </div>
            <div>
              <div class="text-xs text-text-secondary">Annual projection</div>
              <div class="text-lg font-mono font-bold text-status-success">{{ humanCost(result.totals.monthly_savings_usd * 12) }}</div>
            </div>
          </div>
        </template>
        <div v-else class="text-sm text-text-muted py-4">
          Uncheck rules to simulate deletion costs.
        </div>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// RuleCostTable
// ---------------------------------------------------------------------------

const COLUMNS = [
  { key: 'pattern', label: 'Pattern', align: 'left' },
  { key: 'bucket', label: 'Bucket', align: 'left' },
  { key: 'owners', label: 'Owners', align: 'left' },
  { key: 'total_objects', label: 'Objects', align: 'right' },
  { key: 'total_bytes', label: 'Size', align: 'right' },
  { key: 'monthly_cost_usd', label: 'Monthly Cost', align: 'right' },
]

const RuleCostTable = {
  props: { rules: Array },
  emits: ['update:excluded'],
  setup(props, { emit }) {
    const search = ref('')
    const sortKey = ref('monthly_cost_usd')
    const sortDir = ref('desc')
    const excluded = ref(new Set())

    const filteredRules = computed(() => {
      const q = search.value.toLowerCase()
      let rows = props.rules
      if (q) {
        rows = rows.filter(r =>
          r.pattern.toLowerCase().includes(q) ||
          r.owners.toLowerCase().includes(q) ||
          r.bucket.toLowerCase().includes(q)
        )
      }
      return [...rows].sort((a, b) => {
        const av = a[sortKey.value]
        const bv = b[sortKey.value]
        if (typeof av === 'number' && typeof bv === 'number') {
          return sortDir.value === 'asc' ? av - bv : bv - av
        }
        return sortDir.value === 'asc'
          ? String(av).localeCompare(String(bv))
          : String(bv).localeCompare(String(av))
      })
    })

    function toggleSort(key) {
      if (sortKey.value === key) {
        sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
      } else {
        sortKey.value = key
        sortDir.value = 'desc'
      }
    }

    function toggleExcluded(id) {
      const next = new Set(excluded.value)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      excluded.value = next
    }

    watch(excluded, val => emit('update:excluded', val), { deep: true })

    function sortIndicator(key) {
      if (sortKey.value !== key) return ''
      return sortDir.value === 'asc' ? ' \u2191' : ' \u2193'
    }

    return {
      search, sortKey, sortDir, excluded, filteredRules, COLUMNS,
      toggleSort, toggleExcluded, sortIndicator,
      humanBytes, humanCost, humanCount,
    }
  },
  template: `
    <div>
      <div class="mb-3">
        <input v-model="search" type="text"
               placeholder="Filter by pattern, owner, or bucket..."
               class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent" />
      </div>
      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface-sunken">
              <th class="px-3 py-2 w-10"><span class="sr-only">Select</span></th>
              <th v-for="col in COLUMNS" :key="col.key"
                  :class="[
                    'px-3 py-2 text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors',
                    col.align === 'right' ? 'text-right' : 'text-left',
                    sortKey === col.key ? 'text-accent' : 'text-text-secondary',
                  ]"
                  @click="toggleSort(col.key)">
                {{ col.label }}{{ sortIndicator(col.key) }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="rule in filteredRules" :key="rule.id"
                :class="[
                  'border-b border-surface-border-subtle transition-colors',
                  excluded.has(rule.id) ? 'opacity-40' : 'hover:bg-surface-raised',
                ]">
              <td class="px-3 py-2 text-center">
                <input type="checkbox" :checked="!excluded.has(rule.id)"
                       class="accent-accent" @change="toggleExcluded(rule.id)" />
              </td>
              <td class="px-3 py-2 text-[13px] font-mono text-text max-w-xs truncate" :title="rule.pattern">{{ rule.pattern }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ rule.bucket }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ rule.owners }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanCount(rule.total_objects) }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanBytes(rule.total_bytes) }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono font-semibold text-accent">{{ humanCost(rule.monthly_cost_usd) }}</td>
            </tr>
            <tr v-if="filteredRules.length === 0">
              <td :colspan="COLUMNS.length + 1" class="px-3 py-8 text-center text-sm text-text-muted">
                No matching rules
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// Spinner (shared)
// ---------------------------------------------------------------------------

const SPINNER_SVG = `
  <svg class="animate-spin -ml-1 mr-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
`

// ---------------------------------------------------------------------------
// OverviewDashboard
// ---------------------------------------------------------------------------

const OverviewDashboard = {
  components: { RegionCostCard },
  setup() {
    const overview = ref(null)
    const savings = ref(null)
    const loading = ref(true)
    const error = ref(null)

    onMounted(async () => {
      try {
        const [o, s] = await Promise.all([fetchOverview(), fetchSavings()])
        overview.value = o
        savings.value = s
      } catch (e) {
        error.value = e.message || String(e)
      } finally {
        loading.value = false
      }
    })

    return { overview, savings, loading, error, humanBytes, humanCost, humanCount }
  },
  template: `
    <div>
      <div v-if="loading" class="flex items-center justify-center py-24 text-text-muted text-sm">
        ${SPINNER_SVG} Loading storage data...
      </div>
      <div v-else-if="error"
           class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4 text-sm text-status-danger">
        Failed to load data: {{ error }}
      </div>
      <template v-else-if="overview">
        <div class="mb-8">
          <h1 class="text-3xl font-bold text-text tracking-tight">delete-o-tron</h1>
          <div class="mt-2 flex flex-wrap items-baseline gap-6 text-sm">
            <div>
              <span class="text-text-secondary">Total monthly cost: </span>
              <span class="font-mono font-semibold text-accent text-lg">{{ humanCost(overview.totals.total_monthly_cost_usd) }}</span>
            </div>
            <div>
              <span class="text-text-secondary">Total objects: </span>
              <span class="font-mono text-text">{{ humanCount(overview.totals.total_objects) }}</span>
            </div>
            <div>
              <span class="text-text-secondary">Total size: </span>
              <span class="font-mono text-text">{{ humanBytes(overview.totals.total_bytes) }}</span>
            </div>
            <div v-if="savings">
              <span class="text-text-secondary">Potential savings: </span>
              <span class="font-mono font-semibold text-status-success text-lg">{{ humanCost(savings.totals.monthly_savings_usd) }}/mo</span>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <RegionCostCard v-for="region in overview.regions"
                          :key="region.region + region.bucket"
                          :region="region" :discount="overview.discount" />
        </div>

        <div v-if="savings && savings.regions.length > 0"
             class="mt-8 rounded-lg border border-status-success-border bg-status-success-bg p-4">
          <h2 class="text-sm font-semibold text-status-success uppercase tracking-wider mb-3">Potential Savings by Region</h2>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            <div v-for="rs in savings.regions" :key="rs.region"
                 class="flex items-center justify-between text-sm">
              <span class="text-text">{{ rs.region }}</span>
              <div class="text-right">
                <span class="font-mono font-semibold text-status-success">{{ humanCost(rs.monthly_savings_usd) }}/mo</span>
                <span class="text-text-muted ml-2">({{ humanBytes(rs.deletable_bytes) }})</span>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// SavePatterns
// ---------------------------------------------------------------------------

const SavePatterns = {
  components: { RuleCostTable, CostCalculator },
  setup() {
    const rules = ref([])
    const excluded = ref(new Set())
    const loading = ref(true)
    const error = ref(null)

    onMounted(async () => {
      try {
        const resp = await fetchRules()
        rules.value = resp.rules
      } catch (e) {
        error.value = e.message || String(e)
      } finally {
        loading.value = false
      }
    })

    function onExcludedUpdate(ids) {
      excluded.value = ids
    }

    return { rules, excluded, loading, error, onExcludedUpdate }
  },
  template: `
    <div>
      <h1 class="text-2xl font-bold text-text tracking-tight mb-6">Save Patterns</h1>
      <div v-if="loading" class="flex items-center justify-center py-24 text-text-muted text-sm">
        ${SPINNER_SVG} Loading rules...
      </div>
      <div v-else-if="error"
           class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4 text-sm text-status-danger">
        Failed to load rules: {{ error }}
      </div>
      <div v-else class="flex gap-6">
        <div class="flex-1 min-w-0">
          <RuleCostTable :rules="rules" @update:excluded="onExcludedUpdate" />
        </div>
        <div class="w-80 flex-shrink-0">
          <CostCalculator :excluded="excluded" :rules="rules" />
        </div>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

const App = {
  setup() {
    return {}
  },
  template: `
    <div class="min-h-screen bg-surface">
      <header class="border-b border-surface-border bg-surface-sunken">
        <div class="max-w-7xl mx-auto px-6 flex items-center h-14 gap-8">
          <span class="text-lg font-semibold text-text tracking-tight">delete-o-tron</span>
          <nav class="flex items-center gap-1">
            <router-link to="/"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Overview
            </router-link>
            <router-link to="/rules"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/rules' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Save Patterns
            </router-link>
          </nav>
        </div>
      </header>
      <main class="max-w-7xl mx-auto px-6 py-6">
        <router-view />
      </main>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// Router + mount
// ---------------------------------------------------------------------------

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: OverviewDashboard },
    { path: '/rules', component: SavePatterns },
  ],
})

const app = createApp(App)
app.use(router)
app.mount('#app')
