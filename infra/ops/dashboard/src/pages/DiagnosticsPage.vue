<script setup lang="ts">
import { computed, ref } from 'vue'
import { get } from '@/api'
import MetricCard from '@/components/MetricCard.vue'
import PageHeader from '@/components/PageHeader.vue'
import StateChip from '@/components/StateChip.vue'
import { useAutoRefresh } from '@/composables/useAutoRefresh'
import { formatEpochSeconds, formatLogMessage, formatTimestamp } from '@/formatting'
import type { Diagnostics } from '@/types'

const diagnostics = ref<Diagnostics | null>(null)
const error = ref('')

async function refresh() {
  try {
    diagnostics.value = await get<Diagnostics>('/api/diagnostics')
    error.value = ''
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

const latestPoll = computed(() => diagnostics.value?.polls[0])
const dueEscalations = computed(() => diagnostics.value?.escalations.filter((item) => ['pending', 'sending'].includes(item.state)).length ?? 0)

function levelClass(level: string): string {
  if (level === 'ERROR' || level === 'CRITICAL') return 'bg-[#fff0ee] text-[#9a3029]'
  if (level === 'WARNING') return 'bg-[#fff6dc] text-[#785500]'
  return 'bg-[#edf2ec] text-[#4e5d54]'
}

useAutoRefresh(refresh)
</script>

<template>
  <section class="space-y-6">
    <PageHeader section="Runtime" title="Service diagnostics">
      <template #actions>
        <button class="rounded-lg border border-[#cdd2ca] bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-[#f0f2ee]" @click="refresh">Refresh</button>
      </template>
    </PageHeader>

    <div v-if="error" class="rounded-lg border border-[#efb6b2] bg-[#fff1ef] p-3 text-sm text-[#8e2822]">{{ error }}</div>

    <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard label="Last Grafana poll" compact>{{ latestPoll ? formatTimestamp(latestPoll.observed_at) : 'Never' }}</MetricCard>
      <MetricCard label="Firing alerts">{{ latestPoll?.alert_count ?? 0 }}</MetricCard>
      <MetricCard label="Slack due">{{ dueEscalations }}</MetricCard>
      <MetricCard label="Buffered logs">{{ diagnostics?.logs.length ?? 0 }}</MetricCard>
    </div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="flex items-center justify-between border-b border-[#e1e4df] px-5 py-4">
        <h3 class="font-semibold">Grafana polling</h3>
        <span class="text-xs text-[#77817b]">Alertmanager API · newest first</span>
      </div>
      <div v-if="!diagnostics?.polls.length" class="px-5 py-10 text-center text-sm text-[#78817c]">No polls.</div>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="bg-[#f7f8f5] text-xs uppercase tracking-wide text-[#747e77]">
            <tr><th class="px-5 py-3 font-medium">Observed</th><th class="px-5 py-3 font-medium">Poll slot</th><th class="px-5 py-3 text-right font-medium">Firing alerts</th></tr>
          </thead>
          <tbody>
            <tr v-for="poll in diagnostics.polls" :key="poll.poll_slot" class="border-t border-[#edf0eb]">
              <td class="px-5 py-3">{{ formatTimestamp(poll.observed_at) }}</td>
              <td class="px-5 py-3 font-mono text-xs text-[#657169]">{{ formatTimestamp(poll.poll_slot) }}</td>
              <td class="px-5 py-3 text-right font-mono">{{ poll.alert_count }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="flex items-center justify-between border-b border-[#e1e4df] px-5 py-4">
        <h3 class="font-semibold">Slack deliveries</h3>
        <span class="text-xs text-[#77817b]">Durable outbox · newest first</span>
      </div>
      <div v-if="!diagnostics?.escalations.length" class="px-5 py-10 text-center text-sm text-[#78817c]">No agent escalations.</div>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="bg-[#f7f8f5] text-xs uppercase tracking-wide text-[#747e77]">
            <tr><th class="px-5 py-3 font-medium">Created</th><th class="px-5 py-3 font-medium">Severity</th><th class="px-5 py-3 font-medium">Reason</th><th class="px-5 py-3 font-medium">Attempts</th><th class="px-5 py-3 font-medium">State</th></tr>
          </thead>
          <tbody>
            <tr v-for="item in diagnostics.escalations" :key="item.id" class="border-t border-[#edf0eb]">
              <td class="whitespace-nowrap px-5 py-3">{{ formatTimestamp(item.created_at) }}</td>
              <td class="px-5 py-3 font-mono text-xs uppercase">{{ item.severity }}</td>
              <td class="max-w-2xl px-5 py-3">{{ item.reason }}<p v-if="item.last_error" class="mt-1 text-xs text-[#9a3029]">{{ item.last_error }}</p></td>
              <td class="px-5 py-3 font-mono">{{ item.attempts }}</td>
              <td class="px-5 py-3"><StateChip :state="item.state" /></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="flex items-center justify-between border-b border-[#e1e4df] px-5 py-4">
        <h3 class="font-semibold">Process logs</h3>
        <span class="text-xs text-[#77817b]">In-memory · latest 500 · resets on restart</span>
      </div>
      <div v-if="!diagnostics?.logs.length" class="px-5 py-10 text-center text-sm text-[#78817c]">No buffered logs.</div>
      <div v-else class="max-h-[42rem] overflow-auto bg-[#17211c] font-mono text-xs text-[#dce5df]">
        <div v-for="entry in diagnostics.logs" :key="entry.seq" class="grid gap-2 border-b border-white/5 px-4 py-2.5 md:grid-cols-[185px_76px_220px_minmax(0,1fr)]">
          <span class="text-[#93a49a]">{{ formatEpochSeconds(entry.timestamp) }}</span>
          <span><span class="rounded px-1.5 py-0.5 text-[10px] font-semibold" :class="levelClass(entry.level)">{{ entry.level }}</span></span>
          <span class="truncate text-[#a9b9af]" :title="entry.logger_name">{{ entry.logger_name }}</span>
          <pre class="m-0 whitespace-pre-wrap break-words">{{ formatLogMessage(entry.message) }}</pre>
        </div>
      </div>
    </div>
  </section>
</template>
