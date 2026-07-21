<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { get } from '@/api'
import type { Diagnostics } from '@/types'

const diagnostics = ref<Diagnostics | null>(null)
const error = ref('')
let timer: number | undefined

async function refresh() {
  try {
    diagnostics.value = await get<Diagnostics>('/api/diagnostics')
    error.value = ''
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

onMounted(() => {
  void refresh()
  timer = window.setInterval(() => {
    if (!document.hidden) void refresh()
  }, 5_000)
})

onUnmounted(() => {
  if (timer !== undefined) window.clearInterval(timer)
})

function when(value: string | number): string {
  const date = typeof value === 'number' ? new Date(value * 1_000) : new Date(value)
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'medium' }).format(date)
}

function levelClass(level: string): string {
  if (level === 'ERROR' || level === 'CRITICAL') return 'bg-[#fff0ee] text-[#9a3029]'
  if (level === 'WARNING') return 'bg-[#fff6dc] text-[#785500]'
  return 'bg-[#edf2ec] text-[#4e5d54]'
}
</script>

<template>
  <section class="space-y-6">
    <div class="flex items-end justify-between gap-5">
      <div>
        <p class="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-[#657169]">Runtime diagnostics</p>
        <h2 class="text-3xl font-semibold tracking-tight">Polling and process logs</h2>
        <p class="mt-2 max-w-3xl text-sm text-[#68726c]">
          Poll rows are durable in Postgres. Process logs are a bounded convenience view and clear whenever this service instance restarts.
        </p>
      </div>
      <button class="rounded-lg border border-[#cdd2ca] bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-[#f0f2ee]" @click="refresh">
        Refresh
      </button>
    </div>

    <div v-if="error" class="rounded-lg border border-[#efb6b2] bg-[#fff1ef] p-3 text-sm text-[#8e2822]">{{ error }}</div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="flex items-center justify-between border-b border-[#e1e4df] px-5 py-4">
        <h3 class="font-semibold">Recent successful Grafana polls</h3>
        <span class="text-xs text-[#77817b]">Postgres · newest first</span>
      </div>
      <div v-if="!diagnostics?.polls.length" class="px-5 py-10 text-center text-sm text-[#78817c]">No successful polls yet.</div>
      <div v-else class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead class="bg-[#f7f8f5] text-xs uppercase tracking-wide text-[#747e77]">
            <tr><th class="px-5 py-3 font-medium">Observed</th><th class="px-5 py-3 font-medium">Poll slot</th><th class="px-5 py-3 text-right font-medium">Firing alerts</th></tr>
          </thead>
          <tbody>
            <tr v-for="poll in diagnostics.polls" :key="poll.poll_slot" class="border-t border-[#edf0eb]">
              <td class="px-5 py-3">{{ when(poll.observed_at) }}</td>
              <td class="px-5 py-3 font-mono text-xs text-[#657169]">{{ when(poll.poll_slot) }}</td>
              <td class="px-5 py-3 text-right font-mono">{{ poll.alert_count }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="overflow-hidden rounded-xl border border-[#d5dad2] bg-white shadow-[0_2px_8px_rgba(25,38,31,.04)]">
      <div class="flex items-center justify-between border-b border-[#e1e4df] px-5 py-4">
        <h3 class="font-semibold">Live process buffer</h3>
        <span class="text-xs text-[#77817b]">Up to 500 records · newest first</span>
      </div>
      <div v-if="!diagnostics?.logs.length" class="px-5 py-10 text-center text-sm text-[#78817c]">No process logs captured yet.</div>
      <div v-else class="max-h-[42rem] overflow-auto bg-[#17211c] font-mono text-xs text-[#dce5df]">
        <div v-for="entry in diagnostics.logs" :key="entry.seq" class="grid gap-2 border-b border-white/5 px-4 py-2.5 md:grid-cols-[185px_76px_220px_minmax(0,1fr)]">
          <span class="text-[#93a49a]">{{ when(entry.timestamp) }}</span>
          <span><span class="rounded px-1.5 py-0.5 text-[10px] font-semibold" :class="levelClass(entry.level)">{{ entry.level }}</span></span>
          <span class="truncate text-[#a9b9af]" :title="entry.logger_name">{{ entry.logger_name }}</span>
          <pre class="m-0 whitespace-pre-wrap break-words">{{ entry.message }}</pre>
        </div>
      </div>
    </div>
  </section>
</template>
