<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { get, post } from '@/api'
import StateChip from '@/components/StateChip.vue'
import type { CaseDetail, ChatBlock } from '@/types'

const props = defineProps<{ caseId: string }>()
const router = useRouter()
const detail = ref<CaseDetail | null>(null)
const error = ref('')
const message = ref('')
const sending = ref(false)
let timer: number | undefined

async function refresh() {
  try {
    detail.value = await get<CaseDetail>(`/api/cases/${props.caseId}`)
    error.value = ''
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

async function send() {
  if (!message.value.trim() || sending.value) return
  sending.value = true
  try {
    await post(`/api/cases/${props.caseId}/messages`, { text: message.value })
    message.value = ''
    await refresh()
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  } finally {
    sending.value = false
  }
}

async function archive() {
  try {
    await post(`/api/cases/${props.caseId}/archive`)
    await router.push('/')
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  }
}

const visibleChat = computed(() => detail.value?.chat.blocks.filter((block) => ['user_message', 'agent_message', 'thought', 'tool_call'].includes(block.kind)) ?? [])

function blockText(block: ChatBlock): string {
  const text = block.payload.text
  if (typeof text === 'string') {
    if (block.kind === 'user_message' && text.startsWith('You are the Marin ops-expert agent responding to an operational case.')) {
      return 'The ops workflow opened this investigation with the case evidence shown beside the conversation.'
    }
    return text
  }
  const title = block.payload.title
  return typeof title === 'string' ? title : block.kind
}

onMounted(() => {
  void refresh()
  timer = window.setInterval(() => {
    if (!document.hidden) void refresh()
  }, 1_000)
})
onUnmounted(() => {
  if (timer !== undefined) window.clearInterval(timer)
})
</script>

<template>
  <section class="space-y-5">
    <RouterLink to="/" class="text-sm font-medium text-[#356a4d] hover:underline">← Investigation inbox</RouterLink>
    <div v-if="error" class="rounded-lg border border-[#efb6b2] bg-[#fff1ef] p-3 text-sm text-[#8e2822]">{{ error }}</div>
    <div v-if="detail" class="space-y-5">
      <div class="flex flex-wrap items-start justify-between gap-4">
        <div class="min-w-0 max-w-4xl">
          <div class="mb-2 flex items-center gap-2">
            <StateChip :state="detail.case.state" />
            <span class="font-mono text-[11px] text-[#7b847f]">{{ detail.case.id }}</span>
          </div>
          <h2 class="text-2xl font-semibold tracking-tight">{{ detail.case.title }}</h2>
          <p class="mt-2 break-all font-mono text-xs text-[#727b76]">Grafana group: {{ detail.case.group_key }}</p>
        </div>
        <div class="flex gap-2">
          <a v-if="detail.case.loom_session_url" :href="detail.case.loom_session_url" target="_blank" rel="noopener" class="rounded-lg border border-[#cbd2ca] bg-white px-3 py-2 text-sm font-medium hover:bg-[#f1f4ef]">Open full Loom chat ↗</a>
          <button class="rounded-lg border border-[#e2c6c2] bg-white px-3 py-2 text-sm font-medium text-[#8c3931] hover:bg-[#fff2f0]" :disabled="detail.chat.live_turn !== null" @click="archive">Archive</button>
        </div>
      </div>

      <div class="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(420px,0.9fr)]">
        <div class="space-y-5">
          <div class="rounded-xl border border-[#d5dad2] bg-white p-5">
            <div class="mb-4 flex items-center justify-between">
              <h3 class="font-semibold">Grafana evidence</h3>
              <span class="text-xs text-[#7b847f]">{{ detail.signals.length }} fingerprints</span>
            </div>
            <div class="space-y-3">
              <article v-for="signal in detail.signals" :key="signal.fingerprint" class="rounded-lg border border-[#e1e5df] bg-[#fafbf8] p-4">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <div class="flex items-center gap-2">
                    <span class="h-2.5 w-2.5 rounded-full" :class="signal.state === 'firing' ? 'bg-[#e05a40]' : 'bg-[#5d9d65]'" />
                    <span class="font-medium">{{ signal.alert_name }}</span>
                    <span class="rounded bg-[#ecefe9] px-1.5 py-0.5 font-mono text-[10px]">gen {{ signal.signal_generation }}</span>
                  </div>
                  <a v-if="signal.generator_url" :href="signal.generator_url" target="_blank" rel="noopener" class="text-xs font-medium text-[#326d50] hover:underline">Grafana ↗</a>
                </div>
                <p class="mt-2 text-sm text-[#414b45]">{{ signal.summary }}</p>
                <div class="mt-3 flex flex-wrap gap-1.5 font-mono text-[11px] text-[#5d6861]">
                  <span v-if="signal.cluster" class="rounded border border-[#dfe3dc] bg-white px-2 py-1">{{ signal.cluster }}</span>
                  <span v-if="signal.namespace" class="rounded border border-[#dfe3dc] bg-white px-2 py-1">{{ signal.namespace }}</span>
                  <span v-if="signal.object_name" class="rounded border border-[#dfe3dc] bg-white px-2 py-1">{{ signal.object_kind }}/{{ signal.object_name }}</span>
                  <span class="rounded border border-[#dfe3dc] bg-white px-2 py-1">fp {{ signal.fingerprint }}</span>
                </div>
              </article>
            </div>
          </div>

          <div class="rounded-xl border border-[#d5dad2] bg-white p-5">
            <h3 class="mb-3 font-semibold">Workflow timeline</h3>
            <ol class="space-y-2 text-sm">
              <li v-for="event in detail.events" :key="event.id" class="grid grid-cols-[155px_1fr] gap-3 border-l-2 border-[#d9ded7] pl-3">
                <span class="font-mono text-[11px] text-[#7a847e]">{{ new Date(event.created_at).toLocaleString() }}</span>
                <span>{{ event.event_type.split('_').join(' ') }} <span class="text-[#8a928d]">· {{ event.actor }}</span></span>
              </li>
            </ol>
          </div>
        </div>

        <aside class="overflow-hidden rounded-xl border border-[#ccd3cb] bg-[#fbfcfa] shadow-[0_3px_14px_rgba(30,45,36,.06)] lg:sticky lg:top-5 lg:self-start">
          <div class="flex items-center justify-between border-b border-[#dce1da] bg-[#f0f4ee] px-4 py-3">
            <div>
              <h3 class="font-semibold">Ops agent</h3>
              <p class="text-xs text-[#6f7a73]">Read-only Kubernetes and Iris validation</p>
            </div>
            <span v-if="detail.chat.live_turn !== null" class="flex items-center gap-2 text-xs font-semibold text-[#26724a]"><span class="h-2 w-2 animate-pulse rounded-full bg-[#44a66d]" /> Working</span>
            <span v-else class="text-xs text-[#78827c]">Idle</span>
          </div>
          <div class="max-h-[620px] min-h-[360px] space-y-3 overflow-y-auto p-4">
            <div v-if="visibleChat.length === 0" class="py-20 text-center text-sm text-[#7a847e]">The persisted turn is waiting to start.</div>
            <div
              v-for="(block, index) in visibleChat"
              :key="`${block.turn}-${block.seq}-${index}`"
              class="rounded-lg border p-3 text-sm leading-6"
              :class="block.kind === 'agent_message' ? 'border-[#c5d9c6] bg-[#eef7ed]' : block.kind === 'user_message' ? 'border-[#cfd9e5] bg-[#f0f5fb]' : 'border-[#e0e1d8] bg-[#f8f7ee] text-[#667069]'"
            >
              <p class="mb-1 text-[10px] font-semibold uppercase tracking-widest text-[#778179]">{{ block.kind.replace('_', ' ') }}</p>
              <p class="whitespace-pre-wrap">{{ blockText(block) }}</p>
            </div>
          </div>
          <form class="border-t border-[#dce1da] bg-white p-3" @submit.prevent="send">
            <textarea v-model="message" rows="3" class="w-full resize-none rounded-lg border border-[#cdd4cc] p-3 text-sm outline-none focus:border-[#679778] focus:ring-2 focus:ring-[#d9ecdf]" placeholder="Ask a follow-up or add context…" />
            <div class="mt-2 flex items-center justify-between">
              <span class="text-[11px] text-[#858d88]">Queued globally if another agent is active</span>
              <button type="submit" :disabled="sending || !message.trim()" class="rounded-lg bg-[#1c5b3a] px-4 py-2 text-sm font-semibold text-white disabled:opacity-40">{{ sending ? 'Queuing…' : 'Send' }}</button>
            </div>
          </form>
        </aside>
      </div>
    </div>
  </section>
</template>
