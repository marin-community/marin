<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { get } from '@/api'
import MetricCard from '@/components/MetricCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import StateChip from '@/components/StateChip.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { formatTimestamp } from '@/formatting'
import type { CaseRow, Overview } from '@/types'

const overview = ref<Overview | null>(null)
const cases = ref<CaseRow[]>([])
const error = ref('')

async function refresh() {
  try {
    const [nextOverview, nextCases] = await Promise.all([
      get<Overview>('/api/overview'),
      get<{ cases: CaseRow[] }>('/api/cases'),
    ])
    overview.value = nextOverview
    cases.value = nextCases.cases
    error.value = ''
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

const counts = computed(() => overview.value?.case_counts ?? {})
const lastPoll = computed(() => overview.value?.last_poll_at ? formatTimestamp(overview.value.last_poll_at) : 'Never')

useAutoRefresh(refresh)
</script>

<template>
  <section class="space-y-6">
    <PageHeader section="Operations" title="Alert cases">
      <template #actions>
        <button class="rounded-lg border border-[#cdd2ca] bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-[#f0f2ee]" @click="refresh">Refresh</button>
      </template>
    </PageHeader>

    <div v-if="error" class="rounded-lg border border-[#efb6b2] bg-[#fff1ef] p-3 text-sm text-[#8e2822]">{{ error }}</div>

    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="Active">{{ counts.investigating ?? 0 }}</MetricCard>
      <MetricCard label="Queued">{{ counts.pending ?? 0 }}</MetricCard>
      <MetricCard label="Awaiting review">{{ counts.waiting_human ?? 0 }}</MetricCard>
      <MetricCard label="Last Grafana poll" compact>{{ lastPoll }}</MetricCard>
    </div>

    <div v-if="overview?.active_investigation" class="rounded-xl border border-[#b5d6f5] bg-[#eef6ff] p-4">
      <div class="flex items-center gap-3">
        <span class="relative flex h-3 w-3"><span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#388cd3] opacity-50"></span><span class="relative inline-flex h-3 w-3 rounded-full bg-[#2577bd]"></span></span>
        <div class="min-w-0 flex-1">
          <p class="text-xs font-semibold uppercase tracking-wide text-[#3974a6]">Agent running</p>
          <RouterLink :to="`/cases/${overview.active_investigation.case_id}`" class="mt-0.5 block truncate font-medium hover:underline">
            {{ overview.active_investigation.title }}
          </RouterLink>
        </div>
        <a v-if="overview.active_investigation.loom_session_url" :href="overview.active_investigation.loom_session_url" target="_blank" rel="noopener" class="text-xs font-semibold text-[#286fae] hover:underline">Open in Loom ↗</a>
      </div>
    </div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="grid grid-cols-[minmax(0,1fr)_160px_120px] border-b border-[#e1e4df] bg-[#f7f8f5] px-5 py-3 text-[11px] font-medium uppercase tracking-wide text-[#747e77]">
        <span>Case</span><span>Signals</span><span>State</span>
      </div>
      <div v-if="cases.length === 0" class="px-5 py-14 text-center text-sm text-[#78817c]">
        No cases.
      </div>
      <RouterLink
        v-for="item in cases"
        :key="item.id"
        :to="`/cases/${item.id}`"
        class="grid gap-3 border-b border-[#edf0eb] px-5 py-4 last:border-0 hover:bg-[#f8faf6] md:grid-cols-[minmax(0,1fr)_160px_120px] md:items-center"
      >
        <div class="min-w-0">
          <div class="mb-1 flex items-center gap-2">
            <span class="font-mono text-[11px] uppercase tracking-wide text-[#738078]">{{ item.trigger }}</span>
            <span v-if="item.clusters.length" class="rounded bg-[#eef0ec] px-1.5 py-0.5 font-mono text-[11px]">{{ item.clusters.join(', ') }}</span>
          </div>
          <p class="truncate font-medium">{{ item.title }}</p>
          <p v-if="item.summary" class="mt-1 line-clamp-1 text-sm text-[#69736d]">{{ item.summary }}</p>
        </div>
        <div class="text-sm text-[#667169]">
          <span class="font-semibold text-[#26332b]">{{ item.firing_count }}</span> firing / {{ item.signal_count }} signals
        </div>
        <StateChip :state="item.state" />
      </RouterLink>
    </div>
  </section>
</template>
