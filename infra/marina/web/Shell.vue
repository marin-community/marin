<script setup lang="ts">
// The chrome every Marina app renders inside.
//
// One origin serves many apps, each under `/{app}/`, and this is what tells a
// reader which one they are in and how to reach the others. The bar is the
// kernel's; everything below it is the app's, and the app's own navigation goes
// in the `nav` slot beside the switcher rather than in a second bar under it.
//
// `/api/marina/apps` and `/api/marina/me` are the kernel's two GETs.

import { computed, onMounted, ref } from 'vue'

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

/** The local part of the signed-in address, empty when nobody is signed in. */
const who = computed(() => (user.value && user.value !== 'anonymous' ? user.value.split('@')[0] : ''))

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
  const [listed, me] = await Promise.all([
    read<{ apps: App[] }>('/api/marina/apps'),
    read<{ user: string }>('/api/marina/me'),
  ])
  apps.value = listed?.apps ?? []
  user.value = me?.user ?? ''
})
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
    <span v-if="who" class="who" :title="user">{{ who }}</span>
  </header>
  <!-- The page below the bar is the app's: it owns its own width and padding. -->
  <main>
    <slot />
  </main>
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
  margin-left: auto;
  flex: none;
  font: 0.78rem var(--mono);
  color: var(--muted);
  border: 1px solid var(--edge);
  border-radius: 999px;
  padding: 0.05rem 0.55rem;
}

@media (max-width: 40rem) {
  .bar { padding: 0.5rem 0.9rem; }
  .who { margin-left: 0; }
}
</style>
