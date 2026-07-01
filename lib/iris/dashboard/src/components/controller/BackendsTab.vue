<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import { useBackends } from '@/composables/useBackends'
import { useAutoRefresh, DEFAULT_REFRESH_MS } from '@/composables/useAutoRefresh'
import type { BackendSummary, UnroutableJob } from '@/types/rpc'
import InfoCard from '@/components/shared/InfoCard.vue'
import InfoRow from '@/components/shared/InfoRow.vue'
import MetricCard from '@/components/shared/MetricCard.vue'
import ConstraintChip from '@/components/shared/ConstraintChip.vue'
import EmptyState from '@/components/shared/EmptyState.vue'
import LoadingSpinner from '@/components/shared/LoadingSpinner.vue'

// Above this threshold render a compact table instead of the card grid.
const TABLE_THRESHOLD = 8

const { listBackends } = useBackends()

const backendSummaries = ref<BackendSummary[]>([])
const unroutableJobCount = ref(0)
const unroutableSample = ref<UnroutableJob[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

async function refresh() {
  loading.value = true
  error.value = null
  try {
    const resp = await listBackends()
    backendSummaries.value = resp.backends ?? []
    unroutableJobCount.value = resp.unroutableJobCount ?? 0
    unroutableSample.value = resp.unroutableSample ?? []
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(refresh)
useAutoRefresh(refresh, DEFAULT_REFRESH_MS)

const useTable = computed(() => backendSummaries.value.length > TABLE_THRESHOLD)

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

function usersLabel(b: BackendSummary): string {
  if (!b.restricted) return 'all (*)'
  return `restricted (${b.allowedUserCount})`
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
</script>

<template>
  <div class="max-w-7xl mx-auto px-6 py-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-xl font-semibold text-text">
        Backends
        <span v-if="backendSummaries.length" class="ml-2 text-sm font-normal text-text-muted">
          {{ backendSummaries.length }} backend{{ backendSummaries.length !== 1 ? 's' : '' }}
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

    <LoadingSpinner v-if="loading && backendSummaries.length === 0" />

    <EmptyState
      v-else-if="!loading && backendSummaries.length === 0"
      message="No backends registered"
      icon="🖥"
    />

    <!-- Compact table for many backends -->
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
          <tr
            v-for="b in backendSummaries"
            :key="b.backendId"
            class="border-b border-surface-border-subtle hover:bg-surface-raised transition-colors"
          >
            <td class="px-3 py-2 font-mono text-xs">
              <span class="flex items-center gap-1.5">
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
            <td class="px-3 py-2 text-right font-mono tabular-nums">
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
        </tbody>
      </table>
    </div>

    <!-- Card grid for small backend counts -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      <div
        v-for="b in backendSummaries"
        :key="b.backendId"
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
          <InfoRow label="users">{{ usersLabel(b) }}</InfoRow>

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

          <!-- Quick-navigation links -->
          <div class="flex gap-3 pt-1 text-xs">
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
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
