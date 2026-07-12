<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useBackends } from '@/composables/useBackends'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { BackendSummary, PeerSummary, UnroutableJob, ListPeersResponse } from '@/types/rpc'
import { formatRelativeTime } from '@/utils/formatting'
import InfoRow from '@/components/shared/InfoRow.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import ConstraintChip from '@/components/shared/ConstraintChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'
import BackendDetailPanel from '@/components/controller/BackendDetailPanel.vue'

// Above this threshold (counting backends + peers) render a compact table
// instead of the card grid.
const TABLE_THRESHOLD = 8

const { listBackends, listPeers } = useBackends()

// proto3 JSON omits empty repeated fields, so a backend with no scale groups or
// capabilities arrives with those keys absent. Fill them in at the boundary so
// the card/table renderers can treat them as always-present arrays.
function normalizeBackend(b: BackendSummary): BackendSummary {
  return { ...b, capabilities: b.capabilities ?? [], scaleGroups: b.scaleGroups ?? [] }
}

const backendSummaries = ref<BackendSummary[]>([])
const peerSummaries = ref<PeerSummary[]>([])
const unroutableJobCount = ref(0)
const unroutableSample = ref<UnroutableJob[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

// Per-backend detail panels expand on demand; the always-on overview stays
// compact until a backend is opened.
const expanded = ref<Set<string>>(new Set())

function toggleExpanded(backendId: string) {
  const next = new Set(expanded.value)
  if (next.has(backendId)) next.delete(backendId)
  else next.add(backendId)
  expanded.value = next
}

/** A backend has an expandable detail panel when status() authored a variant. */
function hasDetail(b: BackendSummary): boolean {
  return b.detail?.kubernetes != null || b.detail?.worker != null
}

async function refresh() {
  loading.value = true
  error.value = null
  try {
    // Peers and backends are distinct RPCs (distinct ownership); the tab is a
    // display merge only. A peers failure must not blank the backends view.
    const [backendsResp, peersResp] = await Promise.all([
      listBackends(),
      listPeers().catch(() => ({ peers: [] }) as ListPeersResponse),
    ])
    backendSummaries.value = (backendsResp.backends ?? []).map(normalizeBackend)
    unroutableJobCount.value = backendsResp.unroutableJobCount ?? 0
    unroutableSample.value = backendsResp.unroutableSample ?? []
    peerSummaries.value = (peersResp.peers ?? []).map((p) => ({
      ...p,
      backends: (p.backends ?? []).map(normalizeBackend),
    }))
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)

const targetCount = computed(() => backendSummaries.value.length + peerSummaries.value.length)
const useTable = computed(() => targetCount.value > TABLE_THRESHOLD)
const isEmpty = computed(() => targetCount.value === 0)

/**
 * Derive a health dot color class from BackendSummary.capacityHealth + counts.
 * Returns a Tailwind bg-* class string.
 */
function healthDotClass(b: BackendSummary): string {
  const health = b.capacityHealth ?? {}
  const poolCount = Object.values(health).reduce((a, c) => a + c, 0)
  if (poolCount === 0) {
    // No autoscaler data — neutral indicator
    if (b.workerCount === 0 && b.runningTaskCount === 0) return 'bg-text-muted'
    return 'bg-status-success'
  }
  const bad = (health['quota_exceeded'] ?? 0) + (health['backoff'] ?? 0)
  const degraded = health['degraded'] ?? 0
  if (bad > 0) return 'bg-status-danger'
  if (degraded > 0) return 'bg-status-warning'
  return 'bg-status-success'
}

function healthLabel(b: BackendSummary): string {
  const health = b.capacityHealth ?? {}
  const poolCount = Object.values(health).reduce((a, c) => a + c, 0)
  if (poolCount === 0) return b.workerCount > 0 ? 'healthy' : 'no pools'
  const bad = (health['quota_exceeded'] ?? 0) + (health['backoff'] ?? 0)
  const degraded = health['degraded'] ?? 0
  const total = poolCount
  const healthy = total - bad - degraded
  if (bad === 0 && degraded === 0) return `healthy (${total} pools)`
  const parts: string[] = []
  if (bad > 0) parts.push(`${bad} blocked`)
  if (degraded > 0) parts.push(`${degraded} degraded`)
  return `${healthy}/${total} pools · ${parts.join(' · ')}`
}

/** Flatten advertised_attributes into an array of chip strings. */
function deviceChips(b: BackendSummary): string[] {
  const attrs = b.advertisedAttributes ?? {}
  const chips: string[] = []
  for (const [key, list] of Object.entries(attrs)) {
    for (const v of list.values ?? []) {
      chips.push(`${key}=${v}`)
    }
  }
  return chips
}

// -- Peer (federation) helpers: a peer is displayed as one execution target,
// aggregating its forwarded backends' topology. Ownership stays distinct in the
// code and RPCs (ListBackends vs ListPeers); only this tab merges them. --

function peerBackends(p: PeerSummary): BackendSummary[] {
  return p.backends ?? []
}

function peerCaps(p: PeerSummary): string[] {
  const set = new Set<string>()
  for (const b of peerBackends(p)) for (const c of b.capabilities ?? []) set.add(c)
  return [...set]
}

function peerDeviceChips(p: PeerSummary): string[] {
  const set = new Set<string>()
  for (const b of peerBackends(p)) for (const chip of deviceChips(b)) set.add(chip)
  return [...set]
}

function peerWorkerCount(p: PeerSummary): number {
  return peerBackends(p).reduce((a, b) => a + (b.workerCount ?? 0), 0)
}

function peerRunningCount(p: PeerSummary): number {
  return peerBackends(p).reduce((a, b) => a + (b.runningTaskCount ?? 0), 0)
}

function peerPendingCount(p: PeerSummary): number {
  return peerBackends(p).reduce((a, b) => a + (b.pendingTaskCount ?? 0), 0)
}

function peerNeverContacted(p: PeerSummary): boolean {
  return !p.lastContactMs || p.lastContactMs === '0'
}

function peerHealthDotClass(p: PeerSummary): string {
  if (p.reachable) return 'bg-status-success'
  return peerNeverContacted(p) ? 'bg-text-muted' : 'bg-status-danger'
}

function peerHealthLabel(p: PeerSummary): string {
  if (p.reachable) return 'reachable'
  return peerNeverContacted(p) ? 'never contacted' : 'unreachable'
}

function peerLastContact(p: PeerSummary): string {
  const ms = Number(p.lastContactMs ?? '0')
  return ms > 0 ? formatRelativeTime(ms) : 'never'
}
</script>

<template>
  <div class="max-w-7xl mx-auto px-6 py-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-semibold text-text">
        Backends
        <span v-if="!isEmpty" class="ml-2 text-sm font-normal text-text-muted">
          {{ backendSummaries.length }} backend{{ backendSummaries.length !== 1 ? 's' : '' }}<span
            v-if="peerSummaries.length"
          > · {{ peerSummaries.length }} peer{{ peerSummaries.length !== 1 ? 's' : '' }}</span>
        </span>
      </h2>
    </div>

    <!-- Unroutable jobs banner -->
    <div
      v-if="unroutableJobCount > 0"
      class="mb-4 px-4 py-3 rounded-lg border border-status-danger-border bg-status-danger-bg text-sm text-status-danger"
    >
      <span class="font-semibold">{{ unroutableJobCount }} unroutable job{{ unroutableJobCount !== 1 ? 's' : '' }}</span>
      — no backend matches the job's constraints or permits the submitting user.
      <span v-if="unroutableSample.length">
        Sample:
        <RouterLink
          v-for="j in unroutableSample.slice(0, 3)"
          :key="j.jobId"
          :to="'/job/' + encodeURIComponent(j.jobId)"
          class="ml-1 text-accent hover:underline font-mono text-xs"
          :title="j.reason"
        >
          {{ j.jobId.split('/').pop() }}
        </RouterLink>
      </span>
    </div>

    <!-- Error state -->
    <div
      v-if="error"
      class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <LoadingSpinner v-if="loading && isEmpty" />

    <EmptyState
      v-else-if="!loading && isEmpty"
      message="No backends registered"
      icon="🖥"
    />

    <!-- Compact table for many targets -->
    <div v-else-if="useTable" class="rounded-lg border border-surface-border bg-surface overflow-x-auto">
      <table class="w-full border-collapse text-sm">
        <thead>
          <tr class="border-b border-surface-border">
            <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">ID</th>
            <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Kind</th>
            <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Capabilities</th>
            <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Workers</th>
            <th class="px-3 py-2 text-right text-xs font-semibold uppercase tracking-wider text-text-secondary">Tasks</th>
            <th class="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-text-secondary">Health</th>
          </tr>
        </thead>
        <tbody>
          <template
            v-for="b in backendSummaries"
            :key="'backend:' + b.backendId"
          >
            <tr
              class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
              :class="hasDetail(b) ? 'cursor-pointer' : ''"
              @click="hasDetail(b) && toggleExpanded(b.backendId)"
            >
              <td class="px-3 py-2 font-mono text-xs">
                <span class="flex items-center gap-1.5">
                  <span
                    v-if="hasDetail(b)"
                    class="inline-block w-3 text-text-muted transition-transform"
                    :class="expanded.has(b.backendId) ? 'rotate-90' : ''"
                  >▸</span>
                  <span v-else class="inline-block w-3" />
                  <span
                    class="w-2 h-2 rounded-full shrink-0"
                    :class="healthDotClass(b)"
                  />
                  {{ b.name || b.backendId }}
                </span>
              </td>
              <td class="px-3 py-2 text-text-secondary">{{ b.kind }}</td>
              <td class="px-3 py-2">
                <span class="flex flex-wrap gap-1">
                  <span
                    v-for="cap in b.capabilities"
                    :key="cap"
                    class="inline-block rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-xs text-text-secondary"
                  >
                    {{ cap }}
                  </span>
                </span>
              </td>
              <td class="px-3 py-2 text-right font-mono tabular-nums" @click.stop>
                <RouterLink
                  :to="`/fleet?backend=${b.backendId}`"
                  class="text-accent hover:underline"
                >
                  {{ b.workerCount }}
                </RouterLink>
              </td>
              <td class="px-3 py-2 text-right font-mono tabular-nums text-xs">
                {{ b.runningTaskCount }} · {{ b.pendingTaskCount }}
              </td>
              <td class="px-3 py-2 text-xs text-text-secondary">{{ healthLabel(b) }}</td>
            </tr>
            <tr v-if="hasDetail(b) && expanded.has(b.backendId)">
              <td colspan="6" class="p-0">
                <BackendDetailPanel :backend="b" />
              </td>
            </tr>
          </template>

          <!-- Peer rows: not expandable; the ID links inward to the parent's jobs
               list filtered to that cluster (?cluster=). -->
          <tr
            v-for="p in peerSummaries"
            :key="'peer:' + p.peerId"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="px-3 py-2 font-mono text-xs">
              <span class="flex items-center gap-1.5">
                <span class="inline-block w-3" />
                <span class="w-2 h-2 rounded-full shrink-0" :class="peerHealthDotClass(p)" />
                <RouterLink
                  :to="{ path: '/', query: { cluster: p.peerId } }"
                  class="text-accent hover:underline"
                >{{ p.peerId }}</RouterLink>
                <span class="inline-block rounded bg-accent/10 text-accent px-1 text-[10px] font-semibold uppercase tracking-wide">peer</span>
              </span>
            </td>
            <td class="px-3 py-2 text-text-secondary">peer</td>
            <td class="px-3 py-2">
              <span class="flex flex-wrap gap-1">
                <span
                  v-for="cap in peerCaps(p)"
                  :key="cap"
                  class="inline-block rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-xs text-text-secondary"
                >
                  {{ cap }}
                </span>
              </span>
            </td>
            <td class="px-3 py-2 text-right font-mono tabular-nums" @click.stop>
              <RouterLink :to="`/?cluster=${encodeURIComponent(p.peerId)}`" class="text-accent hover:underline">
                {{ peerWorkerCount(p) }}
              </RouterLink>
            </td>
            <td class="px-3 py-2 text-right font-mono tabular-nums text-xs">
              {{ peerRunningCount(p) }} · {{ peerPendingCount(p) }}
            </td>
            <td class="px-3 py-2 text-xs text-text-secondary">{{ peerHealthLabel(p) }}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Card grid: one grid of cards, each a place work can run (backends first,
         then peers). -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      <div
        v-for="b in backendSummaries"
        :key="'backend:' + b.backendId"
        class="rounded-lg border border-surface-border bg-surface"
      >
        <!-- Card header -->
        <div class="px-4 pt-4 pb-2 flex items-center justify-between gap-2">
          <div class="flex items-center gap-2 min-w-0">
            <span
              class="w-2.5 h-2.5 rounded-full shrink-0"
              :class="healthDotClass(b)"
              :title="healthLabel(b)"
            />
            <h3 class="font-semibold text-sm text-text truncate font-mono">
              {{ b.backendId }}
              <span v-if="b.name && b.name !== b.backendId" class="text-text-muted font-normal ml-1">
                · {{ b.name }}
              </span>
            </h3>
          </div>
          <span class="flex gap-1 shrink-0">
            <span
              v-for="cap in b.capabilities"
              :key="cap"
              class="inline-block rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-xs text-text-secondary"
            >
              {{ cap }}
            </span>
          </span>
        </div>

        <div class="px-4 pb-4 space-y-2">
          <InfoRow label="kind">{{ b.kind || '—' }}</InfoRow>

          <!-- Advertised device chips -->
          <div v-if="deviceChips(b).length > 0" class="flex items-start gap-2 text-sm">
            <span class="shrink-0 text-text-secondary">devices</span>
            <span class="flex flex-wrap gap-1">
              <ConstraintChip
                v-for="chip in deviceChips(b)"
                :key="chip"
                :constraint="chip"
              />
            </span>
          </div>

          <div class="grid grid-cols-3 gap-2 pt-1">
            <MetricCard
              :value="b.scaleGroups.length"
              label="Groups"
              size="sm"
            />
            <MetricCard
              :value="b.workerCount"
              label="Workers"
              size="sm"
            />
            <MetricCard
              :value="`${b.runningTaskCount}·${b.pendingTaskCount}`"
              label="Tasks"
              size="sm"
            />
          </div>

          <InfoRow label="capacity">{{ healthLabel(b) }}</InfoRow>

          <!-- Quick-navigation links + detail toggle -->
          <div class="flex items-center gap-3 pt-1 text-xs">
            <RouterLink
              v-if="b.capabilities.includes('workers')"
              :to="`/fleet?backend=${b.backendId}`"
              class="text-accent hover:underline"
            >
              Workers →
            </RouterLink>
            <RouterLink
              v-if="b.scaleGroups.length > 0"
              :to="`/capacity?backend=${b.backendId}`"
              class="text-accent hover:underline"
            >
              Capacity →
            </RouterLink>
            <button
              v-if="hasDetail(b)"
              class="ml-auto text-accent hover:underline"
              @click="toggleExpanded(b.backendId)"
            >
              {{ expanded.has(b.backendId) ? 'Hide details ▾' : 'Show details ▸' }}
            </button>
          </div>
        </div>

        <!-- Expanded detail panel -->
        <BackendDetailPanel v-if="hasDetail(b) && expanded.has(b.backendId)" :backend="b" />
      </div>

      <!-- Peer cards: a federation peer as one execution target. -->
      <div
        v-for="p in peerSummaries"
        :key="'peer:' + p.peerId"
        class="rounded-lg border border-surface-border bg-surface"
      >
        <div class="px-4 pt-4 pb-2 flex items-center justify-between gap-2">
          <div class="flex items-center gap-2 min-w-0">
            <span
              class="w-2.5 h-2.5 rounded-full shrink-0"
              :class="peerHealthDotClass(p)"
              :title="peerHealthLabel(p)"
            />
            <h3 class="font-semibold text-sm text-text truncate font-mono">{{ p.peerId }}</h3>
            <span class="inline-block rounded bg-accent/10 text-accent px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide shrink-0">peer</span>
          </div>
          <span class="flex gap-1 shrink-0">
            <span
              v-for="cap in peerCaps(p)"
              :key="cap"
              class="inline-block rounded bg-surface-sunken px-1.5 py-0.5 font-mono text-xs text-text-secondary"
            >
              {{ cap }}
            </span>
          </span>
        </div>

        <div class="px-4 pb-4 space-y-2">
          <InfoRow label="kind">peer</InfoRow>
          <InfoRow label="status">{{ peerHealthLabel(p) }}</InfoRow>
          <InfoRow label="last contact">{{ peerLastContact(p) }}</InfoRow>

          <!-- Advertised device chips, unioned across the peer's backends -->
          <div v-if="peerDeviceChips(p).length > 0" class="flex items-start gap-2 text-sm">
            <span class="shrink-0 text-text-secondary">devices</span>
            <span class="flex flex-wrap gap-1">
              <ConstraintChip
                v-for="chip in peerDeviceChips(p)"
                :key="chip"
                :constraint="chip"
              />
            </span>
          </div>

          <div class="grid grid-cols-3 gap-2 pt-1">
            <MetricCard :value="peerBackends(p).length" label="Backends" size="sm" />
            <MetricCard :value="peerWorkerCount(p)" label="Workers" size="sm" />
            <MetricCard :value="`${peerRunningCount(p)}·${peerPendingCount(p)}`" label="Tasks" size="sm" />
          </div>

          <InfoRow label="active jobs">{{ p.activeFederatedJobs ?? 0 }}</InfoRow>

          <div class="flex items-center gap-3 pt-1 text-xs">
            <RouterLink :to="{ path: '/', query: { cluster: p.peerId } }" class="ml-auto text-accent hover:underline">
              Jobs →
            </RouterLink>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
