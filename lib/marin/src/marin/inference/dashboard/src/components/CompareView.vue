<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { fetchHealth, fetchInfo } from '../lib/api'
import { comparisonBaseUrl, comparisonTurns } from '../lib/comparison'
import type { PreferenceChoice } from '../lib/comparison'
import { loadComparison, newComparison, saveComparison } from '../lib/comparison_storage'
import type { SamplingParams, ServingInfo } from '../lib/types'
import type { ThinkingMode } from '../lib/chat_template'
import ChatView from './ChatView.vue'

const props = defineProps<{
  model: string
  info: ServingInfo | null
  params: SamplingParams
  system: string
  thinkingMode: ThinkingMode
  customInstructions: string
}>()

const initial = loadComparison() ?? newComparison(props.model, '')
const left = ref(initial.left)
const right = ref(initial.right)
const votes = ref(initial.votes)
const rightUrl = ref('')
const rightBase = ref('')
const rightInfo = ref<ServingInfo | null>(null)
const connectionError = ref('')
const connecting = ref(false)
const sending = ref(false)
const draft = ref('')
const selectedTurn = ref<number | null>(null)
const leftView = ref<InstanceType<typeof ChatView> | null>(null)
const rightView = ref<InstanceType<typeof ChatView> | null>(null)
const completedTurns = computed(() => comparisonTurns(left.value, right.value))

watch(completedTurns, (turns) => {
  selectedTurn.value = turns.length ? turns[turns.length - 1].index : null
}, { immediate: true })

watch(
  () => props.model,
  (model) => {
    if (!model) return
    if (left.value.model && left.value.model !== model) resetComparison(model, rightInfo.value?.model ?? '')
    else left.value.model = model
  },
  { immediate: true },
)

function persist() {
  saveComparison({ left: left.value, right: right.value, votes: votes.value })
}

function resetComparison(leftModel = props.model, rightModel = rightInfo.value?.model ?? '') {
  stop()
  const fresh = newComparison(leftModel, rightModel)
  left.value = fresh.left
  right.value = fresh.right
  votes.value = {}
  persist()
}

async function connect() {
  connecting.value = true
  connectionError.value = ''
  try {
    const base = comparisonBaseUrl(rightUrl.value, window.location.href)
    const info = await fetchInfo(base)
    const health = await fetchHealth(base)
    if (!health.ok) throw new Error('The second model is still loading')
    if (!info.model) throw new Error('The second endpoint did not report a model')
    if ((rightBase.value && rightBase.value !== base) || (right.value.model && right.value.model !== info.model)) {
      resetComparison(props.model, info.model)
    }
    right.value.model = info.model
    rightInfo.value = info
    rightBase.value = base
    persist()
  } catch (error) {
    connectionError.value = error instanceof Error ? error.message : String(error)
  } finally {
    connecting.value = false
  }
}

function stop() {
  leftView.value?.stopStreaming()
  rightView.value?.stopStreaming()
}

function selectPreference(choice: PreferenceChoice) {
  const turnIndex = selectedTurn.value
  if (turnIndex === null || sending.value) return
  const nextVotes = { ...votes.value }
  if (nextVotes[turnIndex] === choice) delete nextVotes[turnIndex]
  else nextVotes[turnIndex] = choice
  votes.value = nextVotes
  persist()
}

async function send() {
  const text = draft.value.trim()
  if (!text || !props.model || !rightInfo.value || sending.value) return
  draft.value = ''
  sending.value = true
  for (const conversation of [left.value, right.value]) {
    conversation.system = props.system
    conversation.thinkingMode = props.thinkingMode
    conversation.customInstructions = props.customInstructions
  }
  try {
    await Promise.all([leftView.value?.send(text), rightView.value?.send(text)])
  } finally {
    sending.value = false
    persist()
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.isComposing || event.key !== 'Enter' || event.shiftKey) return
  event.preventDefault()
  send()
}
</script>

<template>
  <div class="flex min-h-0 flex-1 flex-col">
    <div class="flex flex-wrap items-center gap-2 border-b border-surface-border px-4 py-2">
      <label class="min-w-0 flex flex-1 items-center gap-2 text-xs text-text-secondary">
        Second dashboard URL
        <input
          v-model="rightUrl"
          type="url"
          placeholder="https://iris.example/proxy/t/.../"
          aria-label="Second model dashboard URL"
          class="min-w-40 flex-1 rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm text-text outline-none focus:border-accent"
          @keydown.enter="connect"
        />
      </label>
      <button class="rounded-lg bg-accent px-3 py-1.5 text-sm text-surface disabled:opacity-40" :disabled="connecting || sending || !rightUrl.trim()" @click="connect">
        {{ connecting ? 'Connecting…' : 'Connect' }}
      </button>
      <button class="rounded-lg border border-surface-border px-3 py-1.5 text-sm text-text-secondary hover:text-text" :disabled="sending" @click="resetComparison()">
        New comparison
      </button>
      <p v-if="connectionError" role="alert" class="w-full text-xs text-status-danger">{{ connectionError }}</p>
    </div>
    <div class="grid min-h-0 flex-1 grid-cols-1 overflow-y-auto lg:grid-cols-2 lg:overflow-hidden">
      <section class="flex min-h-80 min-w-0 flex-col border-b border-surface-border lg:min-h-0 lg:border-b-0 lg:border-r" aria-label="First model">
        <div class="truncate border-b border-surface-border bg-surface-raised px-4 py-2 font-mono text-xs text-text-secondary" :title="model">
          {{ model || 'Connecting to first model…' }}
        </div>
        <ChatView
          ref="leftView"
          :conversation="left"
          :params="params"
          :model="model"
          :has-chat-template="info?.has_chat_template ?? true"
          :chat-template-protocol="info?.chat_template_protocol ?? null"
          :streaming="info?.streaming ?? true"
          composer-mode="external"
          @persist="persist"
        />
      </section>
      <section class="flex min-h-80 min-w-0 flex-col lg:min-h-0" aria-label="Second model">
        <div class="truncate border-b border-surface-border bg-surface-raised px-4 py-2 font-mono text-xs text-text-secondary" :title="rightInfo?.model ?? right.model">
          {{ rightInfo?.model || right.model || 'Connect a second model' }}
        </div>
        <ChatView
          v-if="rightInfo"
          ref="rightView"
          :conversation="right"
          :params="params"
          :model="rightInfo.model"
          :has-chat-template="rightInfo.has_chat_template"
          :chat-template-protocol="rightInfo.chat_template_protocol"
          :streaming="rightInfo.streaming"
          :base-url="rightBase"
          composer-mode="external"
          @persist="persist"
        />
        <div v-else class="flex flex-1 items-center justify-center px-6 text-center text-sm text-text-muted">
          Paste another Marin serve dashboard URL above to compare replies.
        </div>
      </section>
    </div>
    <div v-if="completedTurns.length" class="flex flex-wrap items-center gap-2 border-t border-surface-border px-4 py-2 text-xs text-text-secondary">
      <label class="flex items-center gap-2">
        Rate turn
        <select v-model.number="selectedTurn" aria-label="Turn to rate" class="max-w-48 rounded-lg border border-surface-border bg-surface px-2 py-1 text-text">
          <option v-for="turn in completedTurns" :key="turn.index" :value="turn.index">
            {{ turn.index + 1 }}: {{ turn.prompt.slice(0, 50) }}
          </option>
        </select>
      </label>
      <button
        v-for="choice in [{ value: 'left', label: 'Left is better' }, { value: 'right', label: 'Right is better' }, { value: 'tie', label: 'Tie' }, { value: 'both_bad', label: 'Both bad' }] as const"
        :key="choice.value"
        class="rounded-lg border px-2 py-1 disabled:opacity-40"
        :class="selectedTurn !== null && votes[selectedTurn] === choice.value ? 'border-accent bg-accent-subtle text-accent' : 'border-surface-border hover:border-accent hover:text-text'"
        :aria-pressed="selectedTurn !== null && votes[selectedTurn] === choice.value"
        :disabled="sending"
        @click="selectPreference(choice.value)"
      >
        {{ choice.label }}
      </button>
      <span class="text-text-muted">Selections stay in this browser. Click again to clear.</span>
    </div>
    <div class="flex items-end gap-2 border-t border-surface-border px-4 py-3">
      <textarea
        v-model="draft"
        rows="2"
        placeholder="Send the same message to both models…"
        aria-label="Message both models"
        class="max-h-40 min-h-10 flex-1 resize-y rounded-xl border border-surface-border bg-surface-raised px-3.5 py-2.5 text-sm text-text outline-none focus:border-accent"
        @keydown="onKeydown"
      ></textarea>
      <button v-if="sending" class="rounded-xl border border-surface-border px-3 py-2 text-sm text-text-secondary" @click="stop">Stop both</button>
      <button v-else class="rounded-xl bg-accent px-3 py-2 text-sm text-surface disabled:opacity-40" :disabled="!draft.trim() || !model || !rightInfo" @click="send">
        Send to both
      </button>
    </div>
  </div>
</template>
