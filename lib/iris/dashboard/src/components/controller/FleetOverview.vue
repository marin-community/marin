<script setup lang="ts">
import { computed } from 'vue'
import type { ScaleGroupStatus, SliceInfo, RunningTaskBucket } from '@/types/rpc'
import { timestampMs } from '@/utils/formatting'
import { CATEGORICAL_COLORS } from '@/types/status'

// Header "what do we have, where" view: one card per chip type (e.g. v5p-64),
// with a region breakdown and in-use share. Everything is derived from the
// autoscaler groups + the scheduler running buckets the parent already fetches —
// no extra RPCs. Region/chip are parsed from the scale-group name.

const props = defineProps<{
  groups: ScaleGroupStatus[]
  runningBuckets: RunningTaskBucket[]
}>()

interface RegionCount {
  region: string
  count: number
}

interface BandCount {
  band: string
  count: number
}

interface RegionCapacity {
  region: string
  status: 'available' | 'limited' | 'blocked'
  detail: string
}

interface FleetChipSummary {
  chip: string
  total: number
  inUse: number
  avgUptimeMs: number | null
  regions: RegionCount[]
  bands: BandCount[]
  capacity: RegionCapacity[]
}

/** Map workerId → band (lowercased, no prefix) → task count. */
const workerBands = computed<Map<string, Map<string, number>>>(() => {
  const map = new Map<string, Map<string, number>>()
  for (const bucket of props.runningBuckets) {
    if (!bucket.workerId || !bucket.band) continue
    const band = bucket.band.replace(/^PRIORITY_BAND_/, '').toLowerCase()
    if (!map.has(bucket.workerId)) map.set(bucket.workerId, new Map())
    const bands = map.get(bucket.workerId)!
    bands.set(band, (bands.get(band) ?? 0) + bucket.count)
  }
  return map
})

/** True if any VM in the slice is currently running at least one task. */
function sliceInUse(slice: SliceInfo): boolean {
  return (slice.vms ?? []).some(vm => (vm.runningTaskCount ?? 0) > 0)
}

/** Extract chip type + size from scale group name.
 *  e.g. "TPU_V5E_PREEMPTIBLE_16_US_EAST" → "v5e-16"
 *       "TPU_V5P_SERVING_64_US_CENTRAL"  → "v5p-64"
 *       "GPU_A100_8_US_CENTRAL"          → "A100-8"
 */
function chipFromGroupName(name: string): string | null {
  const norm = name.toUpperCase().replace(/-/g, '_')
  const tpuMatch = norm.match(/TPU_(V\d+[A-Z]?)_(?:PREEMPTIBLE|SERVING|ON_DEMAND|RESERVED)_(\d+)/)
  if (tpuMatch) return `${tpuMatch[1].toLowerCase()}-${tpuMatch[2]}`
  const tpuFallback = norm.match(/TPU_(V\d+[A-Z]?)_\w+_(\d+)/)
  if (tpuFallback) return `${tpuFallback[1].toLowerCase()}-${tpuFallback[2]}`
  const gpuMatch = norm.match(/(A100|H100|H200|L4|L40S?|B200)_(\d+)/)
  if (gpuMatch) return `${gpuMatch[1]}-${gpuMatch[2]}`
  if (norm.startsWith('CPU')) return null
  return name
}

/** Extract region from scale group name, dropping the trailing zone letter.
 *  e.g. "TPU_V5E_PREEMPTIBLE_16_US_WEST4_A" → "us-west4"
 */
function regionFromGroupName(name: string): string {
  const norm = name.toUpperCase().replace(/-/g, '_')
  const tpuMatch = norm.match(/TPU_V\d+[A-Z]?_(?:PREEMPTIBLE|SERVING|ON_DEMAND|RESERVED)_\d+_(.+)/)
  if (tpuMatch) {
    let region = tpuMatch[1]
    region = region.replace(/_[A-Z]$/, '')
    return region.toLowerCase().replace(/_/g, '-')
  }
  const fallback = norm.match(/_\d+_(.+)/)
  if (fallback) return fallback[1].toLowerCase().replace(/_/g, '-')
  return 'unknown'
}

const fleetSummary = computed<FleetChipSummary[]>(() => {
  const now = Date.now()
  const chips = new Map<string, { total: number; inUse: number; uptimes: number[]; regions: Map<string, number>; bands: Map<string, number>; capacityByRegion: Map<string, string[]> }>()

  for (const g of props.groups) {
    const chip = chipFromGroupName(g.name)
    if (chip == null) continue
    const region = regionFromGroupName(g.name)
    const entry = chips.get(chip) ?? { total: 0, inUse: 0, uptimes: [] as number[], regions: new Map<string, number>(), bands: new Map<string, number>(), capacityByRegion: new Map<string, string[]>() }

    const readyCount = g.sliceStateCounts?.['ready'] ?? 0
    const readySlices = (g.slices ?? []).filter(s => {
      const state = s.state ?? (s.vms?.length ? 'ready' : '')
      return state === 'ready' || state === 'SLICE_STATE_READY'
    })

    // Count slices (logical machines), not individual VMs.
    const sliceCount = readySlices.length > 0 ? readySlices.length : readyCount
    entry.total += sliceCount
    entry.inUse += readySlices.filter(s => sliceInUse(s)).length
    entry.regions.set(region, (entry.regions.get(region) ?? 0) + sliceCount)

    const statuses = entry.capacityByRegion.get(region) ?? []
    if (g.availabilityStatus) statuses.push(g.availabilityStatus)
    entry.capacityByRegion.set(region, statuses)

    // Assign each in-use slice to a single dominant band (the band with the most
    // task-count across its VMs) so band shares partition slices rather than
    // double-counting slices that host multiple bands.
    for (const slice of readySlices) {
      const sliceBandCounts = new Map<string, number>()
      for (const vm of slice.vms ?? []) {
        if (!vm.workerId) continue
        const bands = workerBands.value.get(vm.workerId)
        if (!bands) continue
        for (const [band, count] of bands) {
          sliceBandCounts.set(band, (sliceBandCounts.get(band) ?? 0) + count)
        }
      }
      let topBand: string | null = null
      let topCount = 0
      for (const [band, count] of sliceBandCounts) {
        if (count > topCount) {
          topBand = band
          topCount = count
        }
      }
      if (topBand) {
        entry.bands.set(topBand, (entry.bands.get(topBand) ?? 0) + 1)
      }
    }

    // Average uptime: use the earliest VM createdAt per slice as the slice uptime.
    for (const slice of readySlices) {
      const vmTimes = (slice.vms ?? [])
        .map(vm => timestampMs(vm.createdAt))
        .filter((ms): ms is number => ms != null && ms > 0)
      if (vmTimes.length > 0) {
        entry.uptimes.push(now - Math.min(...vmTimes))
      }
    }
    chips.set(chip, entry)
  }

  return Array.from(chips.entries())
    .filter(([, c]) => c.total > 0)
    .map(([chip, c]) => ({
      chip,
      total: c.total,
      inUse: c.inUse,
      avgUptimeMs: c.uptimes.length > 0
        ? c.uptimes.reduce((a, b) => a + b, 0) / c.uptimes.length
        : null,
      regions: Array.from(c.regions.entries())
        .map(([region, count]) => ({ region, count }))
        .sort((a, b) => b.count - a.count),
      bands: Array.from(c.bands.entries())
        .map(([band, count]) => ({ band, count }))
        .sort((a, b) => b.count - a.count),
      capacity: Array.from(c.capacityByRegion.entries()).map(([region, statuses]) => {
        if (statuses.includes('quota_exceeded')) {
          return { region, status: 'blocked' as const, detail: 'At Region Quota' }
        }
        if (statuses.includes('at_max_slices')) {
          return { region, status: 'limited' as const, detail: 'At Max Slices' }
        }
        if (statuses.includes('backoff')) {
          return { region, status: 'limited' as const, detail: 'At TRC Capacity' }
        }
        return { region, status: 'available' as const, detail: 'Compute Potentially Available' }
      }).sort((a, b) => {
        const order = { blocked: 0, limited: 1, available: 2 }
        return order[a.status] - order[b.status]
      }),
    }))
    .sort((a, b) => b.total - a.total)
})

/** Stable color index for a region across all chip types. */
const allRegions = computed<Map<string, number>>(() => {
  const regionTotals = new Map<string, number>()
  for (const c of fleetSummary.value) {
    for (const r of c.regions) {
      regionTotals.set(r.region, (regionTotals.get(r.region) ?? 0) + r.count)
    }
  }
  const seen = new Map<string, number>()
  for (const [region] of Array.from(regionTotals.entries()).sort((a, b) => b[1] - a[1])) {
    seen.set(region, seen.size)
  }
  return seen
})

function regionColor(region: string): string {
  const idx = allRegions.value.get(region) ?? 0
  return CATEGORICAL_COLORS[idx % CATEGORICAL_COLORS.length]
}

// Fleet-wide region totals in shared-legend (color) order. One legend above the cards
// names each region's color, so the per-card bars need no repeated legend of their own.
const legendRegions = computed<RegionCount[]>(() => {
  const totals = new Map<string, number>()
  for (const c of fleetSummary.value) {
    for (const r of c.regions) totals.set(r.region, (totals.get(r.region) ?? 0) + r.count)
  }
  return Array.from(totals.entries())
    .map(([region, count]) => ({ region, count }))
    .sort((a, b) => (allRegions.value.get(a.region) ?? 0) - (allRegions.value.get(b.region) ?? 0))
})

function cardTooltip(c: FleetChipSummary): string {
  const lines: string[] = []
  if (c.bands.length > 0) {
    lines.push('Running by band:')
    for (const b of c.bands) {
      const share = c.total > 0 ? Math.round((b.count / c.total) * 100) : 0
      lines.push(`  ${share}% ${b.band}`)
    }
  }
  if (c.capacity.length > 0) {
    if (lines.length > 0) lines.push('')
    lines.push('Capacity:')
    for (const cap of c.capacity) {
      const icon = cap.status === 'available' ? '✓' : cap.status === 'limited' ? '~' : '✗'
      lines.push(`  ${icon} ${cap.region}: ${cap.detail}`)
    }
  }
  return lines.join('\n')
}

function formatUptimeShort(ms: number | null): string {
  if (ms == null) return '-'
  const secs = Math.floor(ms / 1000)
  if (secs < 3600) return `${Math.floor(secs / 60)}m`
  const hours = Math.floor(secs / 3600)
  if (hours < 24) return `${hours}h ${Math.floor((secs % 3600) / 60)}m`
  return `${Math.floor(hours / 24)}d ${hours % 24}h`
}
</script>

<template>
  <section v-if="fleetSummary.length > 0">
    <!-- Heading + one shared region legend; per-card bars reuse these colors. -->
    <div class="flex items-baseline justify-between gap-x-4 gap-y-1 flex-wrap mb-2">
      <h3 class="text-sm font-semibold text-text-secondary uppercase tracking-wider">
        Fleet Overview
      </h3>
      <div class="flex flex-wrap gap-x-3 gap-y-0.5 text-text-muted" style="font-size: clamp(8px, 0.6vw, 11px)">
        <span v-for="r in legendRegions" :key="r.region" class="flex items-center gap-1 whitespace-nowrap">
          <span class="rounded-full inline-block" :style="{ backgroundColor: regionColor(r.region), width: '7px', height: '7px' }" />
          {{ r.region }} {{ r.count }}
        </span>
      </div>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5 gap-2">
      <div
        v-for="c in fleetSummary"
        :key="c.chip"
        class="rounded-lg border border-surface-border bg-surface px-3 py-1.5"
        :title="cardTooltip(c)"
      >
        <!-- Count + chip, inline region bar (regions read from the shared legend;
             exact per-region counts on hover), in-use share. -->
        <div class="flex items-center gap-2" style="font-size: clamp(9px, 0.7vw, 12px)">
          <span class="font-semibold font-mono tabular-nums text-text leading-none" style="font-size: clamp(14px, 1.05vw, 20px)">{{ c.total }}</span>
          <span class="font-medium text-text-secondary uppercase whitespace-nowrap">{{ c.chip }}</span>
          <div class="flex flex-1 min-w-[40px] rounded-full overflow-hidden bg-surface-sunken" style="height: clamp(5px, 0.45vw, 9px)">
            <div
              v-for="r in c.regions"
              :key="r.region"
              class="transition-all"
              :title="`${r.region}: ${r.count}`"
              :style="{ width: (r.count / c.total * 100) + '%', backgroundColor: regionColor(r.region) }"
            />
          </div>
          <span class="tabular-nums whitespace-nowrap" :class="c.total > 0 && c.inUse === c.total ? 'text-status-warning' : 'text-text-muted'">
            {{ c.total > 0 ? Math.round(c.inUse / c.total * 100) : 0 }}% in use
          </span>
        </div>
        <!-- Footer: avg uptime, plus the band mix when it is a genuine mix. -->
        <div class="flex items-center justify-between gap-2 mt-0.5 text-text-muted" style="font-size: clamp(8px, 0.6vw, 11px)">
          <span class="whitespace-nowrap tabular-nums">avg {{ formatUptimeShort(c.avgUptimeMs) }}</span>
          <span v-if="c.bands.length > 1" class="truncate text-right">
            <span v-for="(b, i) in c.bands" :key="b.band">
              <span v-if="i > 0"> · </span>{{ c.total > 0 ? Math.round(b.count / c.total * 100) : 0 }}% {{ b.band }}
            </span>
          </span>
        </div>
      </div>
    </div>
  </section>
</template>
