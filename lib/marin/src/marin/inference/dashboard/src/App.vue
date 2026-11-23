<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue'
import AppHeader from './components/AppHeader.vue'
import ChatView from './components/ChatView.vue'
import CompareView from './components/CompareView.vue'
import CompletionView from './components/CompletionView.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import ReportIssueDialog from './components/ReportIssueDialog.vue'
import SamplingControls from './components/SamplingControls.vue'
import { useServing } from './composables/useServing'
import { createChatShare, fetchChatShare, type ReportInput, submitFeedback } from './lib/api'
import { ThinkingMode } from './lib/chat_template'
import { loadComparison } from './lib/comparison_storage'
import {
  conversationFromSharedChat,
  copySharedChatUrl,
  isSharedChatHash,
  sharedChatIdFromHash,
  sharedChatSnapshot,
  sharedChatUrl,
} from './lib/shared_chat'
import { loadConversations, loadParams, newConversation as createConversation, newId, saveConversations, saveParams } from './lib/storage'
import type { Conversation } from './lib/types'

const { info, status, model } = useServing()

const params = reactive(loadParams())
watch(params, () => saveParams(params))

const conversations = ref<Conversation[]>(loadConversations())
const initialHash = window.location.hash
const sharedHashPresent = isSharedChatHash(initialHash)
const sharedChatId = sharedChatIdFromHash(initialHash)
const active = ref<Conversation>(createConversation(model.value))
if (sharedHashPresent) {
  const cleanUrl = new URL(window.location.href)
  cleanUrl.hash = ''
  window.history.replaceState(null, '', cleanUrl.toString())
}

const sorted = computed(() => [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt))

const mode = ref<'chat' | 'compare' | 'completion'>('chat')
const userPickedMode = ref(false)
const savedCompareSettings = loadComparison()?.left
const compareSettings = reactive({
  system: savedCompareSettings?.system ?? '',
  thinkingMode: savedCompareSettings?.thinkingMode ?? ThinkingMode.TemplateDefault,
  customInstructions: savedCompareSettings?.customInstructions ?? '',
})
const settingsTarget = computed(() => mode.value === 'compare' ? compareSettings : active.value)
const showParams = ref(false)
// Below the md breakpoint the history panel is an overlay drawer.
const showHistory = ref(false)
const shareState = ref<'idle' | 'sharing' | 'copied' | 'ready' | 'failed'>('idle')
const sharedUrl = ref<string | null>(null)
const sharedUrlInput = ref<HTMLInputElement | null>(null)
const shareImportFailed = ref(false)
const reportDialog = ref<InstanceType<typeof ReportIssueDialog> | null>(null)
const chatView = ref<InstanceType<typeof ChatView> | null>(null)
const shareLabel = computed(() => {
  if (shareState.value === 'sharing') return 'Creating link…'
  if (shareState.value === 'copied') return 'Link copied'
  if (shareState.value === 'ready') return 'Link ready'
  if (shareState.value === 'failed') return 'Share failed'
  return 'Share chat'
})
watch(
  () => [active.value.id, active.value.updatedAt],
  () => {
    shareState.value = 'idle'
    sharedUrl.value = null
    shareImportFailed.value = false
  },
)
onMounted(async () => {
  if (!sharedHashPresent) return
  if (!sharedChatId) {
    shareImportFailed.value = true
    return
  }
  try {
    const snapshot = await fetchChatShare(sharedChatId)
    const imported = conversationFromSharedChat(snapshot, newId(), Date.now(), ThinkingMode.TemplateDefault)
    if (!imported) throw new Error('Invalid shared chat snapshot')
    active.value = imported
    persist()
  } catch {
    shareImportFailed.value = true
  }
})

// Base checkpoints without a chat template start in completion mode.
watch(info, (loaded) => {
  if (loaded && !userPickedMode.value) mode.value = loaded.has_chat_template ? 'chat' : 'completion'
})

function pickMode(picked: 'chat' | 'compare' | 'completion') {
  mode.value = picked
  userPickedMode.value = true
}

async function shareConversation() {
  const conversation = active.value
  const updatedAt = conversation.updatedAt
  shareState.value = 'sharing'
  sharedUrl.value = null
  const url = createChatShare(sharedChatSnapshot(conversation)).then((shareId) => sharedChatUrl(window.location.href, shareId))
  const copied = copySharedChatUrl(url)
  try {
    const [link, wasCopied] = await Promise.all([url, copied])
    if (active.value !== conversation || conversation.updatedAt !== updatedAt) return
    sharedUrl.value = link
    shareState.value = wasCopied ? 'copied' : 'ready'
  } catch {
    if (active.value !== conversation || conversation.updatedAt !== updatedAt) return
    shareState.value = 'failed'
  }
}

async function copyShareLink() {
  if (!sharedUrl.value) return
  try {
    await navigator.clipboard.writeText(sharedUrl.value)
    shareState.value = 'copied'
  } catch {
    sharedUrlInput.value?.select()
  }
}

function saveFeedback({ description, reporter, includeConversation }: ReportInput): Promise<string> {
  const conversation = active.value
  return submitFeedback({
    description,
    reporter,
    model: conversation.model || info.value?.model || '',
    conversation: includeConversation ? conversation : null,
  })
}

function persist() {
  const current = active.value
  const alreadySaved = conversations.value.some((conversation) => conversation.id === current.id)
  if (!current.messages.length && !current.pythonTools.trim() && !current.shellWorkspace && !alreadySaved) return
  if (!current.model) current.model = model.value
  if (!alreadySaved) conversations.value.push(current)
  saveConversations(conversations.value)
}

function newConversation() {
  if (active.value.messages.length) persist()
  active.value = createConversation(model.value)
  showHistory.value = false
}

function selectConversation(id: string) {
  const found = conversations.value.find((c) => c.id === id)
  if (found) active.value = found
  showHistory.value = false
}

function removeConversation(id: string) {
  conversations.value = conversations.value.filter((c) => c.id !== id)
  saveConversations(conversations.value)
  if (active.value.id === id) active.value = createConversation(model.value)
}

function clearHistory() {
  conversations.value = []
  saveConversations([])
  active.value = createConversation(model.value)
  showHistory.value = false
}
</script>

<template>
  <div class="flex h-full flex-col">
    <AppHeader :info="info" :status="status" :model="model" />
    <div class="relative flex min-h-0 flex-1">
      <HistoryPanel
        v-if="mode !== 'compare'"
        :conversations="sorted"
        :active-id="active.id"
        :mobile-open="showHistory"
        @select="selectConversation"
        @new="newConversation"
        @remove="removeConversation"
        @clear="clearHistory"
      />
      <main class="@container flex min-w-0 flex-1 flex-col">
        <!-- Breakpoints here use the main column width (@container), not the viewport,
             because the history panel takes part of the screen from md up. -->
        <div class="flex shrink-0 items-center gap-1 border-b border-surface-border px-2 @lg:px-4">
          <button
            v-if="mode !== 'compare'"
            class="mr-1 shrink-0 rounded-lg px-2 py-1 text-text-muted transition-colors hover:text-text md:hidden"
            :class="{ 'bg-surface-sunken text-text': showHistory }"
            title="Conversation history"
            @click="showHistory = !showHistory"
          >
            <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 6h16M4 12h16M4 18h10" />
            </svg>
          </button>
          <div class="flex min-w-0 flex-1 overflow-x-auto">
            <button
              v-for="tab in ['chat', 'compare', 'completion'] as const"
              :key="tab"
              class="shrink-0 border-b-2 px-1.5 py-2 text-sm capitalize transition-colors @lg:px-3"
              :class="
                mode === tab
                  ? 'border-accent font-semibold text-text'
                  : 'border-transparent text-text-muted hover:text-text-secondary'
              "
              @click="pickMode(tab)"
            >
              {{ tab }}
            </button>
          </div>
          <button
            v-if="mode === 'chat'"
            class="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:text-text-secondary disabled:cursor-not-allowed disabled:opacity-40"
            :class="{ 'text-accent': shareState === 'copied', 'text-status-danger': shareState === 'failed' }"
            :disabled="!active.messages.length || shareState === 'sharing'"
            :aria-label="shareLabel"
            title="Copy a short link containing user and assistant messages; hidden prompts and tool details are excluded"
            @click="shareConversation"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <path d="m8.6 10.5 6.8-4M8.6 13.5l6.8 4" />
            </svg>
            <span class="hidden @4xl:inline">{{ shareLabel }}</span>
          </button>
          <button
            v-if="mode === 'chat' && info?.feedback"
            class="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-1 text-xs text-text-muted transition-colors hover:text-text-secondary disabled:cursor-not-allowed disabled:opacity-40"
            :disabled="!active.messages.length || chatView?.busy"
            title="Save a private report about a poor response to Marin storage"
            @click="reportDialog?.open()"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 22V4m0 0h12l-2 4 2 4H4" />
            </svg>
            <span class="hidden @4xl:inline">Report issue</span>
          </button>
          <!-- In narrower columns, max tokens lives in the More options panel instead. -->
          <label class="hidden shrink-0 items-center gap-2 whitespace-nowrap text-xs font-medium text-text-secondary @3xl:flex">
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
            class="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg px-2 py-1 text-xs transition-colors"
            :class="showParams ? 'bg-surface-sunken text-text' : 'text-text-muted hover:text-text-secondary'"
            title="More options"
            :aria-expanded="showParams"
            @click="showParams = !showParams"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6" />
            </svg>
            <span class="hidden @3xl:inline">Temperature {{ params.temperature }}</span>
          </button>
        </div>
        <div
          v-if="shareState === 'ready' && sharedUrl && mode === 'chat'"
          class="flex flex-wrap items-center gap-2 border-b border-surface-border bg-surface-sunken px-4 py-2 text-xs text-text-secondary"
        >
          <span>Link created. Copy it to share this chat:</span>
          <input
            ref="sharedUrlInput"
            :value="sharedUrl"
            aria-label="Shared chat link"
            readonly
            class="min-w-0 flex-1 rounded border border-surface-border bg-surface px-2 py-1 text-text"
            @focus="($event.target as HTMLInputElement).select()"
          />
          <button class="rounded border border-surface-border px-2 py-1 hover:text-text" @click="copyShareLink">Copy link</button>
        </div>
        <div
          v-if="shareState === 'failed' && mode === 'chat'"
          role="alert"
          class="border-b border-status-danger/40 bg-status-danger/10 px-4 py-2 text-xs text-status-danger"
        >
          Could not create a share link. Try again.
        </div>
        <SamplingControls
          v-if="showParams"
          :params="params"
          v-model:system="settingsTarget.system"
          v-model:thinking-mode="settingsTarget.thinkingMode"
          v-model:custom-instructions="settingsTarget.customInstructions"
          :show-chat-controls="mode !== 'completion'"
        />
        <div
          v-if="shareImportFailed && mode === 'chat'"
          role="alert"
          class="border-b border-status-danger/40 bg-status-danger/10 px-4 py-2 text-center text-xs text-status-danger"
        >
          Could not import the shared chat. The link is invalid or was truncated.
        </div>
        <ChatView
          ref="chatView"
          v-if="mode === 'chat'"
          :conversation="active"
          :params="params"
          :model="model"
          :has-chat-template="info ? info.has_chat_template : true"
          :chat-template-protocol="info?.chat_template_protocol ?? null"
          :streaming="info ? info.streaming : true"
          :web-search-available="info?.web_search ?? false"
          @persist="persist"
        />
        <CompareView
          v-show="mode === 'compare'"
          :model="model"
          :info="info"
          :params="params"
          :system="compareSettings.system"
          :thinking-mode="compareSettings.thinkingMode"
          :custom-instructions="compareSettings.customInstructions"
        />
        <ReportIssueDialog ref="reportDialog" :save="saveFeedback" />
        <CompletionView v-if="mode === 'completion'" :params="params" :model="model" :streaming="info ? info.streaming : true" />
      </main>
    </div>
  </div>
</template>
