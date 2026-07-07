<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Overview } from '../types'
import { useQuery } from '../composables/useQuery'
import { DEFAULT_SEED, cnum } from '../utils/formatting'
import SampleTable from './SampleTable.vue'
import SeedControl from './SeedControl.vue'

interface ClusterRow {
  cluster: number
  docs: number
  tokens: number
  q: Record<number, number>
}

const props = defineProps<{ ov: Overview }>()

const nq = computed(() => props.ov.n_quality_buckets)
const qualityBuckets = computed(() => [...Array(nq.value).keys()])

function qualityRange(q: number): string {
  const bucket = props.ov.buckets.find((b) => b.quality_bucket === q)
  return bucket ? bucket.quality_range : `q${q}`
}

const clusterRows = computed<ClusterRow[]>(() => {
  const byCluster = new Map<number, ClusterRow>()
  for (const b of props.ov.buckets) {
    let row = byCluster.get(b.cluster_id)
    if (!row) {
      row = { cluster: b.cluster_id, docs: 0, tokens: 0, q: {} }
      byCluster.set(b.cluster_id, row)
    }
    row.docs += b.total_elements
    row.tokens += b.total_tokens
    row.q[b.quality_bucket] = (row.q[b.quality_bucket] || 0) + b.total_elements
  }
  return [...byCluster.values()]
})

const totals = computed(() => {
  const tot = { docs: 0, tokens: 0, q: {} as Record<number, number> }
  for (const row of clusterRows.value) {
    tot.docs += row.docs
    tot.tokens += row.tokens
    for (const q of qualityBuckets.value) tot.q[q] = (tot.q[q] || 0) + (row.q[q] || 0)
  }
  return tot
})

const sort = ref<{ key: string; dir: 1 | -1 }>({ key: 'docs', dir: -1 })

function setSort(key: string) {
  sort.value =
    sort.value.key === key
      ? { key, dir: sort.value.dir === 1 ? -1 : 1 }
      : { key, dir: key === 'cluster' ? 1 : -1 }
}

const sortedRows = computed(() => {
  const { key, dir } = sort.value
  return [...clusterRows.value].sort((a, b) => {
    let x: number
    let y: number
    if (key === 'cluster' || key === 'docs' || key === 'tokens') {
      x = a[key]
      y = b[key]
    } else {
      x = a.q[+key] || 0
      y = b.q[+key] || 0
    }
    return (x > y ? 1 : x < y ? -1 : 0) * dir
  })
})

function sortIndicator(key: string): string {
  return sort.value.key === key ? (sort.value.dir > 0 ? ' ▲' : ' ▼') : ''
}

// Per-cluster quality mix: stacked share q0..qN (light -> dark = low -> high quality).
function qmixSegments(row: ClusterRow) {
  return qualityBuckets.value.map((q) => {
    const frac = row.docs ? (row.q[q] || 0) / row.docs : 0
    return {
      q,
      title: `q${q} ${qualityRange(q)}: ${(100 * frac).toFixed(0)}%`,
      width: `${(100 * frac).toFixed(1)}%`,
      opacity: (0.2 + (0.8 * q) / Math.max(1, nq.value - 1)).toFixed(2),
    }
  })
}

// Sampling controls. quality '' = any bucket.
const cluster = ref<number>(0)
const quality = ref<string>('')
const sampleSize = ref(12)
const seed = ref(DEFAULT_SEED)

const samples = useQuery<Record<string, unknown>[]>()
const sampleLabel = ref('')
const sampleColumns = ref<string[]>([])
const sampleEmptyText = ref('')

function sampleBucket(c: number, q: string) {
  cluster.value = c
  quality.value = q
  const n = sampleSize.value || 12
  sampleEmptyText.value = 'no documents matched in the sampled sources'
  if (q === '') {
    sampleLabel.value = `cluster ${c} · any quality`
    sampleColumns.value = ['source', 'text']
    void samples.run('store_samples', { cluster: c, n, seed: seed.value })
  } else {
    sampleLabel.value = `cluster ${c} · q${q}`
    sampleColumns.value = ['source', 'score', 'text']
    void samples.run('store_bucket_samples', { cluster: c, quality_bucket: +q, n, seed: seed.value })
  }
}

function sample() {
  sampleBucket(cluster.value, quality.value)
}
</script>

<template>
  <div class="mb-3.5 rounded-lg border border-surface-border bg-surface p-4">
    <div class="mb-2 flex flex-wrap gap-x-5 gap-y-2">
      <div>
        <b class="block text-[17px]">{{ cnum(totals.docs) }}</b>
        <span class="text-xs text-text-muted">documents</span>
      </div>
      <div>
        <b class="block text-[17px]">{{ cnum(totals.tokens) }}</b>
        <span class="text-xs text-text-muted">tokens</span>
      </div>
      <div>
        <b class="block text-[17px]">{{ clusterRows.length }}</b>
        <span class="text-xs text-text-muted">clusters (k={{ ov.cluster_view }})</span>
      </div>
      <div>
        <b class="block text-[17px]">{{ ov.buckets.length }}</b>
        <span class="text-xs text-text-muted">buckets</span>
      </div>
    </div>
    <p class="mb-2 text-sm text-text-muted">
      q0..q{{ nq - 1 }} = quality score ranges
      {{ qualityBuckets.map((q) => qualityRange(q)).join(' · ') }}; click a header to sort
    </p>
    <div class="overflow-auto">
      <table class="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            <th
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-left font-semibold text-text-muted"
              @click="setSort('cluster')"
            >
              cluster{{ sortIndicator('cluster') }}
            </th>
            <th
              v-for="q in qualityBuckets"
              :key="q"
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-right font-semibold text-text-muted"
              @click="setSort(String(q))"
            >
              q{{ q }}{{ sortIndicator(String(q)) }}
            </th>
            <th
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-right font-semibold text-text-muted"
              @click="setSort('docs')"
            >
              docs{{ sortIndicator('docs') }}
            </th>
            <th class="border-b border-surface-border px-2 py-1 text-left font-semibold text-text-muted">
              quality mix
            </th>
            <th
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-right font-semibold text-text-muted"
              @click="setSort('tokens')"
            >
              tokens{{ sortIndicator('tokens') }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in sortedRows" :key="row.cluster">
            <td class="border-b border-surface-border px-2 py-1">c{{ row.cluster }}</td>
            <td
              v-for="q in qualityBuckets"
              :key="q"
              class="cursor-pointer border-b border-surface-border px-2 py-1 text-right hover:bg-accent-subtle"
              :title="`sample cluster ${row.cluster} × q${q}`"
              @click="sampleBucket(row.cluster, String(q))"
            >
              {{ cnum(row.q[q] || 0) }}
            </td>
            <td class="border-b border-surface-border px-2 py-1 text-right font-semibold">{{ cnum(row.docs) }}</td>
            <td class="border-b border-surface-border px-2 py-1">
              <span class="inline-flex w-20 overflow-hidden rounded-sm align-middle">
                <span
                  v-for="seg in qmixSegments(row)"
                  :key="seg.q"
                  :title="seg.title"
                  class="h-[11px] bg-accent"
                  :style="{ width: seg.width, opacity: seg.opacity }"
                ></span>
              </span>
            </td>
            <td class="border-b border-surface-border px-2 py-1 text-right">{{ cnum(row.tokens) }}</td>
          </tr>
          <tr class="border-t-2 border-surface-border font-semibold">
            <td class="px-2 py-1">all</td>
            <td v-for="q in qualityBuckets" :key="q" class="px-2 py-1 text-right">{{ cnum(totals.q[q] || 0) }}</td>
            <td class="px-2 py-1 text-right">{{ cnum(totals.docs) }}</td>
            <td></td>
            <td class="px-2 py-1 text-right">{{ cnum(totals.tokens) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="mb-3 flex flex-wrap items-center gap-2.5">
      <label class="flex items-center gap-1.5 text-sm text-text-secondary">
        cluster
        <select
          v-model.number="cluster"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
        >
          <option v-for="row in sortedRows" :key="row.cluster" :value="row.cluster">
            c{{ row.cluster }} · {{ cnum(row.docs) }} docs
          </option>
        </select>
      </label>
      <label class="flex items-center gap-1.5 text-sm text-text-secondary">
        quality
        <select
          v-model="quality"
          class="rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
        >
          <option value="">any</option>
          <option v-for="q in qualityBuckets" :key="q" :value="String(q)">q{{ q }} {{ qualityRange(q) }}</option>
        </select>
      </label>
      <input
        v-model.number="sampleSize"
        type="number"
        min="1"
        max="50"
        title="sample size"
        class="w-16 rounded border border-surface-border bg-surface px-2 py-1 text-[13px] text-text"
      />
      <SeedControl v-model="seed" @reroll="sample" />
      <button
        class="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-white hover:bg-accent-hover"
        @click="sample"
      >
        sample
      </button>
    </div>

    <p v-if="samples.phase.value === 'idle' && !samples.error.value" class="text-sm text-text-muted">
      pick a cluster + quality (or click a cell in the table above) to sample its documents
    </p>
    <p v-else-if="samples.running.value" class="text-sm text-accent">running query…</p>
    <p v-else-if="samples.error.value" class="text-sm text-status-danger">error: {{ samples.error.value }}</p>
    <template v-else-if="samples.result.value">
      <h4 class="mb-2 font-semibold">
        {{ sampleLabel }} — {{ samples.result.value.length }} sample documents (exact text)
      </h4>
      <SampleTable v-if="samples.result.value.length" :rows="samples.result.value" :columns="sampleColumns" />
      <p v-else class="text-sm text-text-muted">{{ sampleEmptyText }}</p>
    </template>
  </div>
</template>
