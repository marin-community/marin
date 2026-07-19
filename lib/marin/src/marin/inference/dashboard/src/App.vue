<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import AppHeader from './components/AppHeader.vue'
import ChatView from './components/ChatView.vue'
import CompletionView from './components/CompletionView.vue'
import HistoryPanel from './components/HistoryPanel.vue'
import SamplingControls from './components/SamplingControls.vue'
import { useServing } from './composables/useServing'
import { loadConversations, loadParams, newId, saveConversations, saveParams } from './lib/storage'
import type { Conversation } from './lib/types'

const { info, status, model } = useServing()

const params = reactive(loadParams())
watch(params, () => saveParams(params))

const conversations = ref<Conversation[]>(loadConversations())
const active = ref<Conversation>(freshConversation())

const sorted = computed(() => [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt))

const mode = ref<'chat' | 'completion'>('chat')
const userPickedMode = ref(false)
const showParams = ref(false)
// Below the md breakpoint the history panel is an overlay drawer.
const showHistory = ref(false)

// Base checkpoints without a chat template start in completion mode.
watch(info, (loaded) => {
  if (loaded && !userPickedMode.value) mode.value = loaded.has_chat_template ? 'chat' : 'completion'
})

function pickMode(picked: 'chat' | 'completion') {
  mode.value = picked
  userPickedMode.value = true
}

function freshConversation(): Conversation {
  return {
    id: newId(),
    title: '',
    model: model.value,
    system: '',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  }
}

function persist() {
  const current = active.value
  if (!current.messages.length) return
  if (!current.model) current.model = model.value
  if (!conversations.value.some((c) => c.id === current.id)) conversations.value.push(current)
  saveConversations(conversations.value)
}

function newConversation() {
  if (active.value.messages.length) persist()
  active.value = freshConversation()
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
  if (active.value.id === id) active.value = freshConversation()
}

function clearHistory() {
  conversations.value = []
  saveConversations([])
  active.value = freshConversation()
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
          <button
            class="ml-auto flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs transition-colors"
            :class="showParams ? 'bg-surface-sunken text-text' : 'text-text-muted hover:text-text-secondary'"
            title="Sampling parameters"
            @click="showParams = !showParams"
          >
            <svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1 14h6m2-6h6m2 8h6" />
            </svg>
            {{ params.temperature }} · {{ params.maxTokens }}
          </button>
        </div>
        <SamplingControls v-if="showParams" :params="params" v-model:system="active.system" :show-system="mode === 'chat'" />
        <ChatView
          v-if="mode === 'chat'"
          :conversation="active"
          :params="params"
          :model="model"
          :has-chat-template="info ? info.has_chat_template : true"
          @persist="persist"
        />
        <CompletionView v-else :params="params" :model="model" />
      </main>
    </div>
  </div>
</template>
