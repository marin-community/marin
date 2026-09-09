<script setup lang="ts">
// The chrome every Marina app renders inside.
//
// One origin serves many apps, each under `/{app}/`, and this is what tells a
// reader which one they are in and how to reach the others. The bar is the
// kernel's; everything below it is the app's, and the app's own navigation goes
// in the `nav` slot beside the switcher rather than in a second bar under it.
//
// `/api/marina/apps` and `/api/marina/me` are the kernel's two GETs.

import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'

import AgentPanel from './AgentPanel.vue'
import { provideAgentContext } from './agentContext'
import type { AgentPanelConfig } from './loomAgent'

const props = defineProps<{ app: string }>()

/** One app the kernel serves. */
interface App {
  name: string
  title: string
  description: string
  path: string
}

const apps = ref<App[]>([])
const user = ref('')
const agentConfig = ref<AgentPanelConfig>({ enabled: false })
const agentContext = provideAgentContext()
const panelOpen = ref(false)
const launcher = ref<HTMLButtonElement | null>(null)
const modalPanel = ref(false)
let panelMedia: MediaQueryList | null = null
const updatePanelMode = () => (modalPanel.value = panelMedia?.matches ?? false)

/** The local part of the signed-in address, empty when nobody is signed in. */
const who = computed(() => (user.value && user.value !== 'anonymous' ? user.value.split('@')[0] : ''))
const appTitle = computed(() => apps.value.find((candidate) => candidate.name === props.app)?.title ?? props.app)
const canAsk = computed(() => agentConfig.value.enabled && agentContext.value !== null)

// Quiet on failure: the bar is chrome, and an app whose screens work is not
// improved by an error about the switcher above them.
async function read<T>(path: string): Promise<T | undefined> {
  try {
    const response = await fetch(path, { headers: { accept: 'application/json' } })
    if (!response.ok) return undefined
    return (await response.json()) as T
  } catch {
    return undefined
  }
}

onMounted(async () => {
  panelMedia = window.matchMedia('(max-width: 79.99rem)')
  updatePanelMode()
  panelMedia.addEventListener('change', updatePanelMode)
  const [listed, me, configured] = await Promise.all([
    read<{ apps: App[] }>('/api/marina/apps'),
    read<{ user: string }>('/api/marina/me'),
    read<AgentPanelConfig>(`/api/marina/agent/config?app=${encodeURIComponent(props.app)}`),
  ])
  apps.value = listed?.apps ?? []
  user.value = me?.user ?? ''
  agentConfig.value = configured ?? { enabled: false }
})

onBeforeUnmount(() => panelMedia?.removeEventListener('change', updatePanelMode))

function closePanel() {
  panelOpen.value = false
  nextTick(() => launcher.value?.focus())
}
</script>

<template>
  <header class="bar">
    <a class="wordmark" href="/">Marina</a>
    <nav class="apps" aria-label="Apps">
      <a
        v-for="app in apps"
        :key="app.name"
        :href="app.path"
        :title="app.description"
        :aria-current="app.name === props.app ? 'page' : undefined"
        >{{ app.title }}</a
      >
    </nav>
    <nav class="own" aria-label="This app">
      <slot name="nav" />
    </nav>
    <button
      v-if="canAsk"
      ref="launcher"
      class="ask"
      type="button"
      :aria-expanded="panelOpen"
      @click="panelOpen = !panelOpen"
    >
      Ask Marina
    </button>
    <span v-if="who" class="who" :title="user">{{ who }}</span>
  </header>
  <div class="shell-body" :class="{ 'agent-open': panelOpen }">
    <!-- The page below the bar is the app's: it owns its own width and padding. -->
    <main :inert="panelOpen && modalPanel">
      <slot />
    </main>
    <button v-if="panelOpen && modalPanel" class="scrim" type="button" aria-label="Close agent panel" @click="closePanel" />
    <AgentPanel
      v-if="panelOpen && agentContext"
      :app="props.app"
      :app-title="appTitle"
      :config="agentConfig"
      :context="agentContext"
      :modal="modalPanel"
      @close="closePanel"
    />
  </div>
</template>

<style scoped>
.bar {
  display: flex;
  align-items: center;
  gap: 0.5rem 1rem;
  flex-wrap: wrap;
  padding: 0.5rem 1.25rem;
  border-bottom: 1px solid var(--edge);
  background: var(--panel);
}

.wordmark {
  color: var(--ink);
  text-decoration: none;
  font-weight: 600;
  letter-spacing: 0.02em;
  flex: none;
}

/* The switcher scrolls sideways rather than wrapping: on a phone it is one line
 * of names under a fixed wordmark, and a wrapping list would push the app's own
 * navigation off the first screen. */
.apps {
  display: flex;
  gap: 0.9rem;
  font-size: 0.9rem;
  overflow-x: auto;
  scrollbar-width: none;
  padding: 0.15rem 0;
  min-width: 0;
}
.apps::-webkit-scrollbar { display: none; }
.apps a {
  color: var(--muted);
  text-decoration: none;
  white-space: nowrap;
}
.apps a:hover { color: var(--ink); }
.apps a[aria-current='page'] {
  color: var(--ink);
  font-weight: 600;
}

/* The app's own navigation, set off from the switcher by a rule rather than by
 * a second bar, so the two rows of links never read as one.
 *
 * `:slotted` because the links are the app's markup: slot content carries the
 * app's scope and not this component's, so an unqualified `.own a` would miss
 * every one of them. */
.own {
  display: flex;
  gap: 0.9rem;
  font-size: 0.9rem;
  flex-wrap: wrap;
}
.own:not(:empty) {
  border-left: 1px solid var(--edge);
  padding-left: 1rem;
}
.own :slotted(a) {
  color: var(--muted);
  text-decoration: none;
  white-space: nowrap;
}
.own :slotted(a:hover) { color: var(--ink); }
.own :slotted(a[aria-current='page']) {
  color: var(--ink);
  border-bottom: 2px solid var(--mark);
}

.who {
  flex: none;
  font: 0.78rem var(--mono);
  color: var(--muted);
  border: 1px solid var(--edge);
  border-radius: 999px;
  padding: 0.05rem 0.55rem;
}

.ask {
  margin-left: auto;
  border-color: var(--mark);
  background: transparent;
  color: var(--mark);
  padding: 0.25rem 0.65rem;
  font-size: 0.82rem;
}

.shell-body { min-width: 0; }
.shell-body > main { min-width: 0; }
.scrim { display: none; }

@media (min-width: 80rem) {
  .shell-body.agent-open {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 28rem;
    align-items: start;
  }
  .shell-body.agent-open > main { max-height: calc(100vh - 3.35rem); overflow: auto; }
}

@media (max-width: 79.99rem) {
  .scrim {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 30;
    border: 0;
    border-radius: 0;
    background: color-mix(in srgb, var(--ink) 28%, transparent);
  }
}

@media (max-width: 40rem) {
  .bar { padding: 0.5rem 0.9rem; }
  .ask { margin-left: 0; }
}
</style>
