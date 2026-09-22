<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import AppHeader from './components/AppHeader.vue'
import ChatView from './components/ChatView.vue'
import CompletionView from './components/CompletionView.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import SamplingControls from './components/SamplingControls.vue'
import { useServing } from './composables/useServing'
import { ThinkingMode } from './lib/chat_template'
import { isSharedChatHash, sharedChatUrl, sharedConversationFromHash } from './lib/shared_chat'
import { loadConversations, loadParams, newId, saveConversations, saveParams } from './lib/storage'
import type { Conversation } from './lib/types'

const { info, status, model } = useServing()

const params = reactive(loadParams())
watch(params, () => saveParams(params))

const conversations = ref<Conversation[]>(loadConversations())
const sharedHashPresent = isSharedChatHash(window.location.hash)
const importedConversation = sharedConversationFromHash(window.location.hash, newId(), Date.now())
const active = ref<Conversation>(importedConversation ?? freshConversation())
if (sharedHashPresent) {
  const cleanUrl = new URL(window.location.href)
  cleanUrl.hash = ''
  window.history.replaceState(null, '', cleanUrl.toString())
}

const sorted = computed(() => [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt))

const mode = ref<'chat' | 'completion'>('chat')
const userPickedMode = ref(false)
const showParams = ref(false)
// Below the md breakpoint the history panel is an overlay drawer.
const showHistory = ref(false)
const shareState = ref<'idle' | 'copied' | 'failed'>('idle')
const shareImportFailed = ref(sharedHashPresent && !importedConversation)
const shareLabel = computed(() => {
  if (shareState.value === 'copied') return 'Link copied'
  if (shareState.value === 'failed') return 'Copy failed'
  return 'Share chat'
})
watch(
  () => [active.value.id, active.value.updatedAt],
  () => {
    shareState.value = 'idle'
  },
)

// Base checkpoints without a chat template start in completion mode.
watch(info, (loaded) => {
  if (loaded && !userPickedMode.value) mode.value = loaded.has_chat_template ? 'chat' : 'completion'
})

function pickMode(picked: 'chat' | 'completion') {
  mode.value = picked
  userPickedMode.value = true
}

async function shareConversation() {
  try {
    await navigator.clipboard.writeText(sharedChatUrl(window.location.href, active.value))
    shareState.value = 'copied'
  } catch {
    shareState.value = 'failed'
  }
}

function freshConversation(): Conversation {
  return {
    id: newId(),
    title: '',
    model: model.value,
    system: '',
    pythonTools: '',
    shellWorkspace: null,
    thinkingMode: ThinkingMode.TemplateDefault,
    customInstructions: '',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  }
}

function persist() {
  shareImportFailed.value = false
  const current = active.value
  const alreadySaved = conversations.value.some((conversation) => conversation.id === current.id)
  if (!current.messages.length && !current.pythonTools.trim() && !current.shellWorkspace && !alreadySaved) return
  if (!current.model) current.model = model.value
  if (!alreadySaved) conversations.value.push(current)
  saveConversations(conversations.value)
}

function newConversation() {
  if (active.value.messages.length) persist()
  active.value = freshConversation()
  shareImportFailed.value = false
  showHistory.value = false
}

function selectConversation(id: string) {
  const found = conversations.value.find((c) => c.id === id)
  if (found) active.value = found
  shareImportFailed.value = false
  showHistory.value = false
}

function removeConversation(id: string) {
  conversations.value = conversations.value.filter((c) => c.id !== id)
  saveConversations(conversations.value)
  if (active.value.id === id) active.value = freshConversation()
}

function clearHistory() {
  conversations.value = []
  saveConversations([])
  active.value = freshConversation()
  shareImportFailed.value = false
  showHistory.value = false
}
</script>

<template>
  <div class="flex h-full flex-col">
    <AppHeader :info="info" :status="status" :model="model" />
    <div class="relative flex min-h-0 flex-1">
      <HistoryPanel
        :conversations="sorted"
        :active-id="active.id"
        :mobile-open="showHistory"
        @select="selectConversation"
        @new="newConversation"
        @remove="removeConversation"
        @clear="clearHistory"
      />
      <main class="flex min-w-0 flex-1 flex-col">
        <div class="flex shrink-0 items-center gap-1 border-b border-surface-border px-4">
          <button
            class="mr-1 rounded-lg px-2 py-1 text-text-muted transition-colors hover:text-text md:hidden"
            :class="{ 'bg-surface-sunken text-text': showHistory }"
            title="Conversation history"
            @click="showHistory = !showHistory"
          >
            <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 6h16M4 12h16M4 18h10" />
            </svg>
          </button>
          <button
            v-for="tab in ['chat', 'completion'] as const"
            :key="tab"
            class="border-b-2 px-3 py-2 text-sm capitalize transition-colors"
            :class="
              mode === tab
                ? 'border-accent font-semibold text-text'
                : 'border-transparent text-text-muted hover:text-text-secondary'
            "
            @click="pickMode(tab)"
          >
            {{ tab }}
          </button>
          <div class="ml-auto"></div>
          <button
            v-if="mode === 'chat'"
            class="flex items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:text-text-secondary disabled:cursor-not-allowed disabled:opacity-40"
            :class="{ 'text-accent': shareState === 'copied', 'text-status-danger': shareState === 'failed' }"
            :disabled="!active.messages.length"
            title="Copy a link containing the visible chat snapshot; hidden prompts and executable tools are excluded"
            @click="shareConversation"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <path d="m8.6 10.5 6.8-4M8.6 13.5l6.8 4" />
            </svg>
            <span class="hidden sm:inline">{{ shareLabel }}</span>
          </button>
          <label class="flex items-center gap-2 whitespace-nowrap text-xs font-medium text-text-secondary">
            Max tokens
            <input
              v-model.number="params.maxTokens"
              type="number"
              min="1"
              step="1"
              aria-label="Maximum output tokens"
              title="Maximum output tokens generated for each response"
              class="w-20 rounded-lg border border-surface-border bg-surface px-2 py-1 text-sm text-text outline-none transition-colors focus:border-accent"
            />
          </label>
          <button
            class="flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs transition-colors"
            :class="showParams ? 'bg-surface-sunken text-text' : 'text-text-muted hover:text-text-secondary'"
            title="More options"
            @click="showParams = !showParams"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6" />
            </svg>
            <span class="hidden sm:inline">Temperature {{ params.temperature }}</span>
          </button>
        </div>
        <SamplingControls
          v-if="showParams"
          :params="params"
          v-model:system="active.system"
          v-model:thinking-mode="active.thinkingMode"
          v-model:custom-instructions="active.customInstructions"
          :show-chat-controls="mode === 'chat'"
        />
        <div
          v-if="shareImportFailed"
          role="alert"
          class="border-b border-status-danger/40 bg-status-danger/10 px-4 py-2 text-center text-xs text-status-danger"
        >
          Could not import the shared chat. The link is invalid or was truncated.
        </div>
        <ChatView
          v-if="mode === 'chat'"
          :conversation="active"
          :params="params"
          :model="model"
          :has-chat-template="info ? info.has_chat_template : true"
          :chat-template-protocol="info?.chat_template_protocol ?? null"
          :streaming="info ? info.streaming : true"
          @persist="persist"
        />
        <CompletionView v-else :params="params" :model="model" :streaming="info ? info.streaming : true" />
      </main>
    </div>
  </div>
</template>
