// Capture token from URL and persist to localStorage
;(function() {
  const params = new URLSearchParams(window.location.search)
  const token = params.get('token')
  if (token) {
    localStorage.setItem('storage_token', token)
    // Strip token from URL to keep it clean
    params.delete('token')
    const clean = params.toString()
    const url = window.location.pathname + (clean ? '?' + clean : '')
    window.history.replaceState({}, '', url)
  }
})()

import { createApp, ref, computed, watch, onMounted } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { humanBytes, humanCost, humanCount } from './format.js'
import {
  fetchOverview, fetchSavings, fetchRules, fetchSimulate, fetchDeleteEstimate, fetchExplore,
  fetchDeleteRules, createDeleteRule, removeDeleteRule,
  createProtectRule, removeProtectRule,
  fetchUnifiedExplore, triggerSync, fetchSyncStatus,
} from './api.js'

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
  emits: ['update:excluded', 'remove'],
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
      emit,
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
              <th class="px-3 py-2 w-16"><span class="sr-only">Actions</span></th>
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
              <td class="px-3 py-2 text-right">
                <button @click="emit('remove', rule.id)"
                        class="text-xs text-text-muted hover:text-status-danger transition-colors">
                  Remove
                </button>
              </td>
            </tr>
            <tr v-if="filteredRules.length === 0">
              <td :colspan="COLUMNS.length + 2" class="px-3 py-8 text-center text-sm text-text-muted">
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

const MARIN_BUCKET_OPTIONS = ['*']

const SavePatterns = {
  components: { RuleCostTable, CostCalculator },
  setup() {
    const rules = ref([])
    const excluded = ref(new Set())
    const loading = ref(true)
    const error = ref(null)
    const addError = ref(null)
    const bucketOptions = ref(MARIN_BUCKET_OPTIONS)
    const newRule = ref({ pattern: '', bucket: '*', owners: '', reasons: '' })

    onMounted(async () => {
      try {
        const [rulesResp, overviewResp] = await Promise.all([fetchRules(), fetchOverview()])
        rules.value = rulesResp.rules
        const buckets = overviewResp.regions.map(r => r.bucket)
        bucketOptions.value = ['*', ...buckets]
      } catch (e) {
        error.value = e.message || String(e)
      } finally {
        loading.value = false
      }
    })

    function onExcludedUpdate(ids) {
      excluded.value = ids
    }

    async function addRule() {
      addError.value = null
      try {
        await createProtectRule(newRule.value)
        newRule.value = { pattern: '', bucket: '*', owners: '', reasons: '' }
        const resp = await fetchRules()
        rules.value = resp.rules
      } catch (e) {
        addError.value = e.message || String(e)
      }
    }

    async function removeRule(id) {
      try {
        await removeProtectRule(id)
        const resp = await fetchRules()
        rules.value = resp.rules
      } catch (e) {
        error.value = e.message || String(e)
      }
    }

    return { rules, excluded, loading, error, addError, newRule, bucketOptions, onExcludedUpdate, addRule, removeRule }
  },
  template: `
    <div>
      <h1 class="text-2xl font-bold text-text tracking-tight mb-6">Protect Rules</h1>

      <!-- Add rule form -->
      <div class="rounded-lg border border-surface-border bg-surface-raised p-4 mb-6">
        <h2 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">Add Protect Rule</h2>
        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <label class="text-xs text-text-muted block mb-1">Bucket</label>
            <select v-model="newRule.bucket"
                    class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text focus:outline-none focus:border-accent">
              <option v-for="b in bucketOptions" :key="b" :value="b">{{ b }}</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-text-muted block mb-1">Pattern (SQL LIKE)</label>
            <input v-model="newRule.pattern" placeholder="e.g. checkpoints/best/%"
                   class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent font-mono" />
          </div>
          <div>
            <label class="text-xs text-text-muted block mb-1">Owners</label>
            <input v-model="newRule.owners" placeholder="e.g. alice,bob"
                   class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent" />
          </div>
          <div>
            <label class="text-xs text-text-muted block mb-1">Reasons</label>
            <input v-model="newRule.reasons" placeholder="Why keep these?"
                   class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent" />
          </div>
        </div>
        <div class="mt-3 flex items-center gap-3">
          <button @click="addRule" :disabled="!newRule.pattern"
                  class="px-4 py-2 text-sm font-medium rounded bg-accent text-white hover:bg-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
            Add Rule
          </button>
          <span v-if="addError" class="text-sm text-status-danger">{{ addError }}</span>
        </div>
      </div>

      <div v-if="loading" class="flex items-center justify-center py-24 text-text-muted text-sm">
        ${SPINNER_SVG} Loading rules...
      </div>
      <div v-else-if="error"
           class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4 text-sm text-status-danger">
        Failed to load rules: {{ error }}
      </div>
      <div v-else class="flex gap-6">
        <div class="flex-1 min-w-0">
          <RuleCostTable :rules="rules" @update:excluded="onExcludedUpdate" @remove="removeRule" />
        </div>
        <div class="w-80 flex-shrink-0">
          <CostCalculator :excluded="excluded" :rules="rules" />
        </div>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// Explorer
// ---------------------------------------------------------------------------

const Explorer = {
  setup() {
    const buckets = ref([])
    const selectedBucket = ref('')
    const pathStack = ref([])  // array of prefix strings for breadcrumb nav
    const loading = ref(false)
    const error = ref(null)
    const result = ref(null)
    const sortKey = ref('monthly_cost_usd')
    const sortDir = ref('desc')

    const currentPrefix = computed(() => {
      return pathStack.value.length > 0 ? pathStack.value[pathStack.value.length - 1] : ''
    })

    const breadcrumbs = computed(() => {
      const parts = [{ label: '/', prefix: '' }]
      if (pathStack.value.length === 0) return parts
      const full = pathStack.value[pathStack.value.length - 1]
      const segments = full.split('/').filter(Boolean)
      let acc = ''
      for (const seg of segments) {
        acc += seg + '/'
        parts.push({ label: seg, prefix: acc })
      }
      return parts
    })

    const sortedEntries = computed(() => {
      if (!result.value) return []
      return [...result.value.entries].sort((a, b) => {
        const av = a[sortKey.value]
        const bv = b[sortKey.value]
        if (typeof av === 'number' && typeof bv === 'number') {
          return sortDir.value === 'asc' ? av - bv : bv - av
        }
        return sortDir.value === 'asc'
          ? String(av ?? '').localeCompare(String(bv ?? ''))
          : String(bv ?? '').localeCompare(String(av ?? ''))
      })
    })

    function toggleSort(key) {
      if (sortKey.value === key) {
        sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
      } else {
        sortKey.value = key
        sortDir.value = key === 'name' ? 'asc' : 'desc'
      }
    }

    function sortIndicator(key) {
      if (sortKey.value !== key) return ''
      return sortDir.value === 'asc' ? ' \\u2191' : ' \\u2193'
    }

    async function load() {
      if (!selectedBucket.value) return
      loading.value = true
      error.value = null
      try {
        result.value = await fetchExplore(selectedBucket.value, currentPrefix.value)
      } catch (e) {
        error.value = e.message || String(e)
      } finally {
        loading.value = false
      }
    }

    function navigate(prefix) {
      pathStack.value = [...pathStack.value, prefix]
      load()
    }

    function navigateTo(prefix) {
      // Used by breadcrumbs — truncate stack to this prefix
      if (prefix === '') {
        pathStack.value = []
      } else {
        const idx = pathStack.value.indexOf(prefix)
        if (idx >= 0) {
          pathStack.value = pathStack.value.slice(0, idx + 1)
        } else {
          pathStack.value = [prefix]
        }
      }
      load()
    }

    onMounted(async () => {
      try {
        const overview = await fetchOverview()
        buckets.value = overview.regions.map(r => ({ bucket: r.bucket, region: r.region }))
        if (buckets.value.length > 0) {
          selectedBucket.value = buckets.value[0].bucket
          load()
        }
      } catch (e) {
        error.value = e.message || String(e)
      }
    })

    watch(selectedBucket, () => {
      pathStack.value = []
      load()
    })

    return {
      buckets, selectedBucket, pathStack, loading, error, result,
      currentPrefix, breadcrumbs, sortedEntries,
      sortKey, sortDir, toggleSort, sortIndicator,
      navigate, navigateTo, load,
      humanBytes, humanCost, humanCount,
    }
  },
  template: `
    <div>
      <h1 class="text-2xl font-bold text-text tracking-tight mb-6">Explore</h1>

      <div class="flex items-center gap-4 mb-4">
        <select v-model="selectedBucket"
                class="px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text focus:outline-none focus:border-accent">
          <option v-for="b in buckets" :key="b.bucket" :value="b.bucket">
            {{ b.region }} ({{ b.bucket }})
          </option>
        </select>
      </div>

      <div class="flex items-center gap-1 mb-4 text-sm">
        <template v-for="(crumb, idx) in breadcrumbs" :key="crumb.prefix">
          <span v-if="idx > 0" class="text-text-muted">/</span>
          <button @click="navigateTo(crumb.prefix)"
                  :class="[
                    'px-1.5 py-0.5 rounded transition-colors',
                    idx === breadcrumbs.length - 1
                      ? 'text-text font-semibold'
                      : 'text-accent hover:text-accent-hover hover:bg-surface-raised'
                  ]">
            {{ crumb.label }}
          </button>
        </template>
      </div>

      <div v-if="loading" class="flex items-center gap-2 text-sm text-text-muted py-8">
        <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        Loading...
      </div>
      <div v-else-if="error"
           class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4 text-sm text-status-danger">
        {{ error }}
      </div>
      <template v-else-if="result">
        <div v-if="result.type === 'buckets'" class="mb-2 text-xs text-text-muted">
          Too many children — showing lexicographic ranges
        </div>

        <div class="overflow-x-auto rounded-lg border border-surface-border">
          <table class="w-full border-collapse">
            <thead>
              <tr class="border-b border-surface-border bg-surface-sunken">
                <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                    :class="sortKey === 'name' ? 'text-accent' : 'text-text-secondary'"
                    @click="toggleSort('name')">
                  Name{{ sortIndicator('name') }}
                </th>
                <th v-if="result.type === 'buckets'"
                    class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">
                  Dirs
                </th>
                <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                    :class="sortKey === 'objects' ? 'text-accent' : 'text-text-secondary'"
                    @click="toggleSort('objects')">
                  Objects{{ sortIndicator('objects') }}
                </th>
                <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                    :class="sortKey === 'bytes' ? 'text-accent' : 'text-text-secondary'"
                    @click="toggleSort('bytes')">
                  Size{{ sortIndicator('bytes') }}
                </th>
                <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                    :class="sortKey === 'monthly_cost_usd' ? 'text-accent' : 'text-text-secondary'"
                    @click="toggleSort('monthly_cost_usd')">
                  Cost/mo{{ sortIndicator('monthly_cost_usd') }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="entry in sortedEntries" :key="entry.name"
                  class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors">
                <td class="px-3 py-2 text-[13px] font-mono text-text max-w-md">
                  <button v-if="entry.prefix"
                          @click="navigate(entry.prefix)"
                          class="text-accent hover:text-accent-hover hover:underline text-left">
                    {{ entry.name }}
                  </button>
                  <span v-else class="text-text-secondary">{{ entry.name }}</span>
                </td>
                <td v-if="result.type === 'buckets'" class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">
                  {{ humanCount(entry.child_count) }}
                </td>
                <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanCount(entry.objects) }}</td>
                <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanBytes(entry.bytes) }}</td>
                <td class="px-3 py-2 text-[13px] text-right font-mono font-semibold text-accent">{{ humanCost(entry.monthly_cost_usd) }}</td>
              </tr>
              <tr v-if="sortedEntries.length === 0">
                <td :colspan="result.type === 'buckets' ? 5 : 4" class="px-3 py-8 text-center text-sm text-text-muted">
                  Empty directory
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// DeleteRulesManager
// ---------------------------------------------------------------------------

const DeleteRulesManager = {
  setup() {
    const rules = ref([])
    const newRule = ref({ pattern: '', storage_class: null, description: '' })
    const loading = ref(false)

    const totalObjects = computed(() => rules.value.reduce((s, r) => s + (r.total_objects || 0), 0))
    const totalCost = computed(() => rules.value.reduce((s, r) => s + (r.monthly_cost_usd || 0), 0))
    const totalBytesHuman = computed(() => {
      const bytes = rules.value.reduce((s, r) => s + (r.total_bytes || 0), 0)
      return humanBytes(bytes)
    })

    async function load() {
      loading.value = true
      try {
        const data = await fetchDeleteRules()
        rules.value = data.rules || []
      } finally {
        loading.value = false
      }
    }

    async function addRule() {
      await createDeleteRule(newRule.value)
      newRule.value = { pattern: '', storage_class: null, description: '' }
      await load()
    }

    async function removeRule(id) {
      await removeDeleteRule(id)
      await load()
    }

    onMounted(load)

    return { rules, newRule, loading, totalObjects, totalCost, totalBytesHuman, addRule, removeRule, humanBytes, humanCost }
  },
  template: `
    <div class="space-y-6">
      <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold text-text tracking-tight">Delete Rules</h1>
      </div>

      <!-- Add rule form -->
      <div class="rounded-lg border border-surface-border bg-surface-raised p-4">
        <h2 class="text-sm font-semibold text-text-secondary uppercase tracking-wider mb-3">Add Delete Rule</h2>
        <div class="flex gap-3 items-end flex-wrap">
          <div class="flex-1 min-w-40">
            <label class="text-xs text-text-muted block mb-1">Pattern (SQL LIKE)</label>
            <input v-model="newRule.pattern" placeholder="e.g. checkpoints/%"
                   class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent font-mono" />
          </div>
          <div>
            <label class="text-xs text-text-muted block mb-1">Storage Class</label>
            <select v-model="newRule.storage_class"
                    class="px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text focus:outline-none focus:border-accent">
              <option :value="null">Any</option>
              <option value="STANDARD">STANDARD</option>
              <option value="NEARLINE">NEARLINE</option>
              <option value="COLDLINE">COLDLINE</option>
              <option value="ARCHIVE">ARCHIVE</option>
            </select>
          </div>
          <div class="flex-1 min-w-40">
            <label class="text-xs text-text-muted block mb-1">Description</label>
            <input v-model="newRule.description" placeholder="Reason for deletion"
                   class="w-full px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text placeholder-text-muted focus:outline-none focus:border-accent" />
          </div>
          <button @click="addRule" :disabled="!newRule.pattern || loading"
                  class="px-4 py-2 text-sm font-medium rounded bg-status-danger text-white hover:opacity-80 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed">
            {{ loading ? 'Loading...' : 'Add Rule' }}
          </button>
        </div>
      </div>

      <!-- Rules table -->
      <div class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface-sunken">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Pattern</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Storage Class</th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Description</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Objects</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Size</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">$/mo</th>
              <th class="px-3 py-2 w-16"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="rule in rules" :key="rule.id"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors">
              <td class="px-3 py-2 text-[13px] font-mono text-text max-w-xs truncate" :title="rule.pattern">{{ rule.pattern }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ rule.storage_class || 'Any' }}</td>
              <td class="px-3 py-2 text-[13px] text-text-secondary">{{ rule.description }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ (rule.total_objects || 0).toLocaleString() }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanBytes(rule.total_bytes || 0) }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono font-semibold text-status-danger">\${{ (rule.monthly_cost_usd || 0).toFixed(2) }}</td>
              <td class="px-3 py-2 text-right">
                <button @click="removeRule(rule.id)" :disabled="loading"
                        class="text-xs text-text-muted hover:text-status-danger transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                  Remove
                </button>
              </td>
            </tr>
            <tr v-if="!rules.length">
              <td colspan="7" class="px-3 py-8 text-center text-sm text-text-muted">
                No delete rules configured. Add one above.
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Totals -->
      <div v-if="rules.length"
           class="rounded-lg border border-status-danger-border bg-status-danger-bg p-4">
        <div class="text-xs text-text-secondary uppercase tracking-wider mb-1">Total deletion impact</div>
        <div class="text-lg font-mono font-semibold text-status-danger">
          {{ totalObjects.toLocaleString() }} objects &middot; {{ totalBytesHuman }} &middot; \${{ totalCost.toFixed(2) }}/mo
        </div>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// UnifiedExplorer
// ---------------------------------------------------------------------------

const UnifiedExplorer = {
  setup() {
    const prefix = ref('')
    const entries = ref([])
    const responseType = ref('children')
    const loading = ref(false)
    const sort = ref({ key: 'name', desc: false })

    // Bucket & storage class filters
    const buckets = ref([])
    const selectedBucket = ref('')
    const STORAGE_CLASSES = ['STANDARD', 'NEARLINE', 'COLDLINE', 'ARCHIVE']
    const selectedStorageClass = ref('')

    const breadcrumbs = computed(() => {
      if (!prefix.value) return []
      const parts = prefix.value.replace(/\/$/, '').split('/')
      return parts.map((p, i) => ({
        name: p,
        prefix: parts.slice(0, i + 1).join('/') + '/',
      }))
    })

    const sortedEntries = computed(() => {
      const key = sort.value.key
      const mult = sort.value.desc ? -1 : 1
      return [...entries.value].sort((a, b) => {
        // For bucketed sub-prefixes, group by status first
        if (responseType.value === 'buckets') {
          const sa = a.status_order ?? 99
          const sb = b.status_order ?? 99
          if (sa !== sb) return sa - sb
        }
        // Sort by user-selected column (lex for TLDs, within-status for buckets)
        if (typeof a[key] === 'string') return mult * a[key].localeCompare(b[key])
        return mult * ((a[key] || 0) - (b[key] || 0))
      })
    })

    async function load() {
      loading.value = true
      try {
        const data = await fetchUnifiedExplore(prefix.value, selectedBucket.value, selectedStorageClass.value)
        entries.value = data.entries || []
        responseType.value = data.type || 'children'
      } finally {
        loading.value = false
      }
    }

    function navigate(newPrefix) {
      prefix.value = newPrefix
      load()
    }

    function sortBy(key) {
      if (sort.value.key === key) sort.value = { key, desc: !sort.value.desc }
      else sort.value = { key, desc: true }
    }

    function statusClass(status) {
      return {
        keep: 'bg-status-success-bg text-status-success',
        delete: 'bg-status-danger-bg text-status-danger',
        mixed: 'bg-status-warning-bg text-status-warning',
        unmatched: 'bg-surface-sunken text-text-muted',
      }[status] || 'bg-surface-sunken text-text-muted'
    }

    function sortIndicator(key) {
      if (sort.value.key !== key) return ''
      return sort.value.desc ? ' \u2193' : ' \u2191'
    }

    watch(selectedBucket, () => { prefix.value = ''; load() })
    watch(selectedStorageClass, load)

    onMounted(async () => {
      try {
        const overview = await fetchOverview()
        buckets.value = overview.regions.map(r => ({ bucket: r.bucket, region: r.region }))
        if (buckets.value.length > 0) {
          selectedBucket.value = buckets.value[0].bucket
          return // watcher will trigger load
        }
      } catch (_) {}
      load()
    })

    return {
      prefix, entries, responseType, loading, sort, breadcrumbs, sortedEntries,
      buckets, selectedBucket, STORAGE_CLASSES, selectedStorageClass,
      navigate, sortBy, statusClass, sortIndicator, humanBytes, humanCost,
    }
  },
  template: `
    <div>
      <h1 class="text-2xl font-bold text-text tracking-tight mb-6">Storage Explorer</h1>

      <!-- Filters -->
      <div class="flex items-center gap-4 mb-4">
        <select v-model="selectedBucket"
                class="px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text focus:outline-none focus:border-accent">
          <option value="">All buckets</option>
          <option v-for="b in buckets" :key="b.bucket" :value="b.bucket">
            {{ b.region }} ({{ b.bucket }})
          </option>
        </select>
        <select v-model="selectedStorageClass"
                class="px-3 py-2 text-sm rounded border border-surface-border bg-surface-sunken text-text focus:outline-none focus:border-accent">
          <option value="">All classes</option>
          <option v-for="sc in STORAGE_CLASSES" :key="sc" :value="sc">{{ sc }}</option>
        </select>
      </div>

      <!-- Breadcrumb -->
      <div class="flex items-center gap-1 mb-4 text-sm">
        <button @click="navigate('')"
                class="px-1.5 py-0.5 rounded text-accent hover:text-accent-hover hover:bg-surface-raised transition-colors">
          /
        </button>
        <template v-for="(seg, i) in breadcrumbs" :key="i">
          <span class="text-text-muted">/</span>
          <button @click="navigate(seg.prefix)"
                  :class="[
                    'px-1.5 py-0.5 rounded transition-colors',
                    i === breadcrumbs.length - 1
                      ? 'text-text font-semibold'
                      : 'text-accent hover:text-accent-hover hover:bg-surface-raised',
                  ]">
            {{ seg.name }}
          </button>
        </template>
      </div>

      <div v-if="loading" class="flex items-center gap-2 text-sm text-text-muted py-8">
        ${SPINNER_SVG} Loading...
      </div>
      <div v-else class="overflow-x-auto rounded-lg border border-surface-border">
        <table class="w-full border-collapse">
          <thead>
            <tr class="border-b border-surface-border bg-surface-sunken">
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                  :class="sort.key === 'name' ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy('name')">
                Name{{ sortIndicator('name') }}
              </th>
              <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Status</th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                  :class="sort.key === 'objects' ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy('objects')">
                Objects{{ sortIndicator('objects') }}
              </th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                  :class="sort.key === 'bytes' ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy('bytes')">
                Size{{ sortIndicator('bytes') }}
              </th>
              <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider cursor-pointer select-none hover:text-text transition-colors"
                  :class="sort.key === 'monthly_cost_usd' ? 'text-accent' : 'text-text-secondary'"
                  @click="sortBy('monthly_cost_usd')">
                Cost/mo{{ sortIndicator('monthly_cost_usd') }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in sortedEntries" :key="entry.name"
                class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors">
              <td class="px-3 py-2 text-[13px] font-mono text-text max-w-md">
                <button v-if="entry.prefix" @click="navigate(entry.prefix)"
                        class="text-accent hover:text-accent-hover hover:underline text-left">
                  {{ entry.name }}
                </button>
                <span v-else class="text-text-secondary">{{ entry.name }}</span>
              </td>
              <td class="px-3 py-2">
                <span v-if="entry.status"
                      :class="[statusClass(entry.status), 'text-xs font-medium px-2 py-0.5 rounded']">
                  {{ entry.status }}
                </span>
              </td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ (entry.objects || 0).toLocaleString() }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono text-text-secondary">{{ humanBytes(entry.bytes || 0) }}</td>
              <td class="px-3 py-2 text-[13px] text-right font-mono font-semibold text-accent">
                <span v-if="entry.kept_cost != null">{{ humanCost(entry.kept_cost) }}</span>
                <span v-else>{{ humanCost(entry.monthly_cost_usd || 0) }}</span>
              </td>
            </tr>
            <tr v-if="!sortedEntries.length && !loading">
              <td colspan="5" class="px-3 py-8 text-center text-sm text-text-muted">No entries</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// SyncStatus
// ---------------------------------------------------------------------------

const SyncStatus = {
  setup() {
    const status = ref({})

    function formatTime(iso) {
      if (!iso) return ''
      return new Date(iso).toLocaleTimeString()
    }

    async function load() {
      try { status.value = await fetchSyncStatus() } catch (_) {}
    }

    async function sync() {
      await triggerSync()
      await load()
    }

    onMounted(() => {
      load()
      setInterval(load, 30000)
    })

    return { status, formatTime, sync }
  },
  template: `
    <div class="flex items-center gap-2 text-xs text-text-muted">
      <span v-if="status.syncing" class="text-status-warning">Syncing...</span>
      <span v-else-if="status.last_sync">Last sync: {{ formatTime(status.last_sync) }}</span>
      <span v-else>No sync</span>
      <button @click="sync" :disabled="status.syncing"
              class="text-accent hover:underline disabled:opacity-50">
        Sync
      </button>
    </div>
  `,
}

// ---------------------------------------------------------------------------
// App shell
// ---------------------------------------------------------------------------

const App = {
  components: { SyncStatus },
  setup() {
    return {}
  },
  template: `
    <div class="min-h-screen bg-surface">
      <header class="border-b border-surface-border bg-surface-sunken">
        <div class="max-w-7xl mx-auto px-6 flex items-center h-14 gap-8">
          <span class="text-lg font-semibold text-text tracking-tight">delete-o-tron</span>
          <nav class="flex items-center gap-1 flex-1">
            <router-link to="/"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Overview
            </router-link>
            <router-link to="/rules"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/rules' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Protect Rules
            </router-link>
            <router-link to="/delete-rules"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/delete-rules' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Delete Rules
            </router-link>
            <router-link to="/explore"
              :class="['px-3 py-1.5 rounded text-sm font-medium transition-colors',
                $route.path === '/explore' ? 'bg-surface-raised text-text' : 'text-text-secondary hover:text-text hover:bg-surface-raised']">
              Explorer
            </router-link>
          </nav>
          <SyncStatus />
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
    { path: '/delete-rules', component: DeleteRulesManager },
    { path: '/explore', component: UnifiedExplorer },
  ],
})

const app = createApp(App)
app.use(router)
app.mount('#app')
