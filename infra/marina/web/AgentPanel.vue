<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

import Prose from './Prose.vue'
import type { AppAgentContext } from './agentContext'
import {
  LoomAgentClient,
  pageContext,
  visibleUserText,
  type AgentPanelConfig,
  type AgentSession,
  type ChatBlock,
} from './loomAgent'

const props = defineProps<{
  app: string
  appTitle: string
  config: AgentPanelConfig
  context: AppAgentContext
  modal: boolean
}>()
const emit = defineEmits<{ close: [] }>()

const SESSION_MAP_KEY = 'marina.agent.sessions.v2'
const DRAFT_STORAGE_KEY = 'marina.agent.drafts.v2'
const SESSION_LIMIT = 20

interface StoredSessions {
  version: 2
  username: string
  loomOrigin: string
  sessions: Record<string, { sessionId: string; lastOpenedAt: string }>
}

interface StoredDrafts {
  version: 2
  username: string
  loomOrigin: string
  drafts: Record<string, string>
}

const client = new LoomAgentClient(
  {
    origin: props.config.origin ?? '',
    profile: props.config.profile ?? '',
    repository: props.config.repository ?? '',
  },
  props.appTitle,
)
const state = ref<'connecting' | 'signed-out' | 'ready' | 'error'>('connecting')
const username = ref('')
const session = ref<AgentSession | null>(null)
const blocks = ref<ChatBlock[]>([])
const draft = ref('')
const error = ref('')
const live = ref(false)
const liveText = ref('')
const liveTool = ref<{ title: string; status: string } | null>(null)
const sending = ref(false)
const continuityUnavailable = ref(false)
const composer = ref<HTMLTextAreaElement | null>(null)
const panel = ref<HTMLElement | null>(null)
let closeStream: (() => void) | null = null

const contextKey = computed(() => `${props.app}:${props.context.contextKey}`)
const orderedBlocks = computed(() => [...blocks.value].sort((a, b) => a.turn - b.turn || a.seq - b.seq))

function readStorage<T>(storage: Storage, key: string): T | null {
  try {
    const value = storage.getItem(key)
    return value ? (JSON.parse(value) as T) : null
  } catch {
    continuityUnavailable.value = true
    return null
  }
}

function writeStorage(storage: Storage, key: string, value: unknown): void {
  try {
    storage.setItem(key, JSON.stringify(value))
  } catch {
    continuityUnavailable.value = true
  }
}

function storedSessions(): StoredSessions {
  const stored = readStorage<StoredSessions>(localStorage, SESSION_MAP_KEY)
  if (
    stored?.version === 2 &&
    stored.username === username.value &&
    stored.loomOrigin === props.config.origin
  ) {
    return stored
  }
  const empty: StoredSessions = {
    version: 2,
    username: username.value,
    loomOrigin: props.config.origin ?? '',
    sessions: {},
  }
  writeStorage(localStorage, SESSION_MAP_KEY, empty)
  return empty
}

function mappedSession(): string | null {
  return storedSessions().sessions[contextKey.value]?.sessionId ?? null
}

function rememberSession(sessionId: string): void {
  const stored = storedSessions()
  stored.sessions[contextKey.value] = { sessionId, lastOpenedAt: new Date().toISOString() }
  const entries = Object.entries(stored.sessions).sort(([, a], [, b]) => b.lastOpenedAt.localeCompare(a.lastOpenedAt))
  stored.sessions = Object.fromEntries(entries.slice(0, SESSION_LIMIT))
  writeStorage(localStorage, SESSION_MAP_KEY, stored)
}

function forgetSession(): void {
  const stored = storedSessions()
  delete stored.sessions[contextKey.value]
  writeStorage(localStorage, SESSION_MAP_KEY, stored)
}

function storedDrafts(): StoredDrafts {
  const stored = readStorage<StoredDrafts>(sessionStorage, DRAFT_STORAGE_KEY)
  if (
    stored?.version === 2 &&
    stored.username === username.value &&
    stored.loomOrigin === props.config.origin
  ) {
    return stored
  }
  const empty: StoredDrafts = {
    version: 2,
    username: username.value,
    loomOrigin: props.config.origin ?? '',
    drafts: {},
  }
  writeStorage(sessionStorage, DRAFT_STORAGE_KEY, empty)
  return empty
}

function restoreDraft(): void {
  draft.value = storedDrafts().drafts[contextKey.value] ?? ''
}

function saveDraft(): void {
  if (!username.value) return
  const stored = storedDrafts()
  stored.drafts[contextKey.value] = draft.value
  writeStorage(sessionStorage, DRAFT_STORAGE_KEY, stored)
}

function upsertBlock(block: ChatBlock): void {
  const key = `${block.turn}:${block.seq}`
  const index = blocks.value.findIndex((candidate) => `${candidate.turn}:${candidate.seq}` === key)
  if (index < 0) blocks.value = [...blocks.value, block]
  else blocks.value[index] = block
  if (block.kind === 'agent_message') liveText.value = ''
  if (block.kind === 'tool_call') liveTool.value = null
}

async function loadSnapshot(sessionId: string): Promise<void> {
  const snapshot = await client.snapshot(sessionId)
  blocks.value = snapshot.blocks
  live.value = snapshot.live_turn !== null
  if (!live.value) {
    liveText.value = ''
    liveTool.value = null
  }
}

async function bind(next: AgentSession): Promise<void> {
  closeStream?.()
  session.value = next
  blocks.value = []
  liveText.value = ''
  liveTool.value = null
  closeStream = client.stream(next.id, {
    event(name, value) {
      if (name === 'block') upsertBlock(value as ChatBlock)
      if (name === 'delta') {
        const delta = value as { kind?: string; text?: string }
        if (delta.kind === 'agent_message') liveText.value += delta.text ?? ''
      }
      if (name === 'tool') {
        const tool = value as { title?: string; status?: string }
        liveTool.value = { title: tool.title ?? 'Agent activity', status: tool.status ?? 'running' }
      }
      if (name === 'turn') {
        const turn = value as { state?: string }
        live.value = turn.state === 'started'
        if (turn.state === 'ended') void loadSnapshot(next.id)
      }
      if (name === 'resync') {
        liveText.value = ''
        liveTool.value = null
        void loadSnapshot(next.id)
      }
    },
    open() {
      void loadSnapshot(next.id)
    },
    error() {
      if (state.value === 'ready') error.value = 'Reconnecting to Loom…'
    },
  })
  await loadSnapshot(next.id)
  error.value = ''
}

async function connect(): Promise<void> {
  state.value = 'connecting'
  error.value = ''
  try {
    const identity = await client.auth()
    if (!identity.authenticated || !identity.username) {
      state.value = 'signed-out'
      return
    }
    username.value = identity.username
    restoreDraft()
    const storedId = mappedSession()
    if (storedId) {
      try {
        const restored = await client.get(storedId)
        if (restored.profile === props.config.profile && restored.protocol === 'acp') await bind(restored)
        else forgetSession()
      } catch (cause) {
        if (String(cause).includes('404')) forgetSession()
        else throw cause
      }
    }
    state.value = 'ready'
    await nextTick()
    composer.value?.focus()
  } catch (cause) {
    state.value = 'error'
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

async function submit(message = draft.value): Promise<void> {
  const text = message.trim()
  if (!text || sending.value || live.value) return
  sending.value = true
  error.value = ''
  try {
    const context = pageContext(props.app, props.context)
    if (!session.value) {
      const launched = await client.launch(context, text)
      rememberSession(launched.id)
      await bind(launched)
    } else {
      await client.prompt(session.value.id, context, text)
      await loadSnapshot(session.value.id)
    }
    draft.value = ''
    saveDraft()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  } finally {
    sending.value = false
  }
}

async function stop(): Promise<void> {
  if (!session.value) return
  try {
    await client.interrupt(session.value.id)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

async function openInLoom(): Promise<void> {
  if (!session.value) return
  try {
    const resolved = await client.sessionUrl(session.value.id)
    const url = new URL(resolved.url)
    const loopback = url.protocol === 'http:' && ['127.0.0.1', '::1'].includes(url.hostname)
    if (url.origin !== props.config.origin || (url.protocol !== 'https:' && !loopback)) {
      throw new Error('Loom returned an invalid session URL')
    }
    window.open(url.href, '_blank', 'noopener')
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

async function answerPermission(block: ChatBlock, optionId: string): Promise<void> {
  if (!session.value) return
  try {
    await client.answerPermission(session.value.id, String(block.payload.request_id), optionId)
    await loadSnapshot(session.value.id)
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause)
  }
}

function permissionOptions(block: ChatBlock): Array<{ option_id: string; name: string; kind?: string }> {
  const options = Array.isArray(block.payload.options)
    ? (block.payload.options as Array<{ option_id: string; name: string; kind?: string }>)
    : []
  const rank = (kind: string | undefined) => (kind === 'reject_once' || kind === 'reject_always' ? 0 : 1)
  return [...options].sort((left, right) => rank(left.kind) - rank(right.kind))
}

function newConversation(): void {
  closeStream?.()
  closeStream = null
  forgetSession()
  session.value = null
  blocks.value = []
  live.value = false
  liveText.value = ''
  liveTool.value = null
  nextTick(() => composer.value?.focus())
}

function payloadText(block: ChatBlock): string {
  const text = typeof block.payload.text === 'string' ? block.payload.text : ''
  return block.kind === 'user_message' ? visibleUserText(text) : text
}

function keydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') emit('close')
  if (event.key === 'Tab' && props.modal && panel.value) {
    const focusable = [...panel.value.querySelectorAll<HTMLElement>('a[href], button:not(:disabled), textarea:not(:disabled)')]
    if (!focusable.length) return
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') void submit()
}

onMounted(() => {
  window.addEventListener('keydown', keydown)
  panel.value?.focus()
  void connect()
})
onBeforeUnmount(() => {
  saveDraft()
  closeStream?.()
  window.removeEventListener('keydown', keydown)
})
</script>

<template>
  <aside
    ref="panel"
    class="agent-panel"
    aria-labelledby="agent-heading"
    :role="modal ? 'dialog' : 'complementary'"
    :aria-modal="modal ? 'true' : undefined"
    tabindex="-1"
  >
    <header class="agent-header">
      <div>
        <strong id="agent-heading">Ask Marina</strong>
        <span>{{ context.label }}</span>
      </div>
      <nav aria-label="Agent conversation">
        <button v-if="session" class="quiet" type="button" @click="newConversation">New</button>
        <button v-if="session" class="quiet" type="button" @click="openInLoom">Open in Loom</button>
        <button ref="closeButton" class="quiet close" type="button" aria-label="Close agent panel" @click="emit('close')">×</button>
      </nav>
    </header>

    <div v-if="state === 'connecting'" class="agent-state">Connecting to Loom…</div>
    <div v-else-if="state === 'signed-out'" class="agent-state">
      <p>Sign in to Loom to ask about this page.</p>
      <a class="button primary" :href="`${config.origin}/`" target="_blank" rel="noreferrer">Sign in to Loom</a>
      <button type="button" @click="connect">Try again</button>
    </div>
    <div v-else-if="state === 'error'" class="agent-state">
      <p class="problem">{{ error }}</p>
      <button type="button" @click="connect">Try again</button>
    </div>

    <template v-else>
      <div class="transcript" aria-live="polite">
        <div v-if="!session" class="empty">
          <p>Ask about the current chart. Marina can inspect the plan through its read-only API.</p>
          <button v-for="starter in config.starters" :key="starter" type="button" @click="submit(starter)">
            {{ starter }}
          </button>
        </div>

        <template v-for="block in orderedBlocks" :key="`${block.turn}:${block.seq}`">
          <article v-if="block.kind === 'user_message'" class="message user-message">
            <span>You</span>
            <p>{{ payloadText(block) }}</p>
          </article>
          <article v-else-if="block.kind === 'agent_message'" class="message agent-message">
            <span>Marina</span>
            <Prose :text="payloadText(block)" />
          </article>
          <details v-else-if="block.kind === 'tool_call'" class="activity">
            <summary>{{ block.payload.title || 'Agent activity' }} · {{ block.payload.status || 'complete' }}</summary>
          </details>
          <section v-else-if="block.kind === 'permission_request' && !block.payload.outcome" class="permission">
            <strong>{{ block.payload.title || 'Permission requested' }}</strong>
            <button
              v-for="option in permissionOptions(block)"
              :key="option.option_id"
              type="button"
              @click="answerPermission(block, option.option_id)"
            >
              {{ option.name }}
            </button>
          </section>
        </template>

        <article v-if="liveText" class="message agent-message streaming">
          <span>Marina</span>
          <Prose :text="liveText" />
        </article>
        <details v-if="liveTool" class="activity live-activity" open>
          <summary>{{ liveTool.title }} · {{ liveTool.status }}</summary>
        </details>
        <p v-else-if="live" class="working">Thinking…</p>
      </div>

      <p v-if="error" class="connection-state">{{ error }}</p>
      <p v-if="continuityUnavailable" class="connection-state">Browser storage is unavailable; this conversation will not restore after refresh.</p>
      <form class="composer" @submit.prevent="submit()">
        <textarea
          ref="composer"
          v-model="draft"
          aria-label="Ask about this page"
          placeholder="Ask about this page…"
          rows="2"
          :disabled="sending"
          @input="saveDraft"
        />
        <button v-if="live" type="button" @click="stop">Stop</button>
        <button v-else class="primary" type="submit" :disabled="sending || !draft.trim()">Send</button>
      </form>
    </template>
  </aside>
</template>

<style scoped>
.agent-panel {
  display: flex;
  flex-direction: column;
  min-width: 0;
  height: calc(100vh - 3.35rem);
  background: var(--paper);
  border-left: 1px solid var(--edge);
  box-shadow: -12px 0 32px color-mix(in srgb, var(--ink) 10%, transparent);
  z-index: 40;
}

.agent-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.8rem 0.9rem;
  border-bottom: 1px solid var(--edge);
}
.agent-header > div { min-width: 0; display: grid; }
.agent-header span { color: var(--muted); font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agent-header nav { display: flex; align-items: center; gap: 0.2rem; }
.agent-header button { padding: 0.2rem 0.4rem; font-size: 0.78rem; }
.close { font-size: 1.2rem !important; line-height: 1; }

.transcript { flex: 1; overflow-y: auto; padding: 1rem; display: flex; flex-direction: column; gap: 1rem; }
.empty { margin: auto 0; display: grid; gap: 0.65rem; color: var(--muted); }
.empty button { text-align: left; color: var(--ink); background: var(--panel); }
.message { display: grid; gap: 0.25rem; }
.message > span { font: 0.7rem var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
.message p { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.user-message { margin-left: 1.6rem; padding: 0.65rem 0.75rem; border-radius: var(--radius); background: var(--panel); border: 1px solid var(--edge); }
.agent-message { padding-right: 0.5rem; }
.streaming { opacity: 0.9; }
.activity { color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--edge); padding-top: 0.5rem; }
.live-activity summary, .working { color: var(--mark); }
.working { margin: 0; font-size: 0.85rem; }
.permission { border: 1px solid var(--warn); border-radius: var(--radius); padding: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.permission strong { flex-basis: 100%; }
.agent-state { margin: auto; max-width: 20rem; padding: 1rem; text-align: center; display: grid; gap: 0.75rem; justify-items: center; }
.agent-state p { margin: 0; }
.connection-state { margin: 0; padding: 0.35rem 0.8rem; color: var(--muted); font-size: 0.78rem; border-top: 1px solid var(--edge); }
.composer { display: flex; align-items: flex-end; gap: 0.5rem; padding: 0.75rem; border-top: 1px solid var(--edge); background: var(--panel); }
.composer textarea { flex: 1; resize: none; max-height: 12rem; }

@media (max-width: 79.99rem) {
  .agent-panel { position: fixed; top: 0; right: 0; width: min(32rem, 100vw); height: 100vh; }
}

@media (max-width: 47.99rem) {
  .agent-panel { width: 100vw; border-left: 0; }
}
</style>
