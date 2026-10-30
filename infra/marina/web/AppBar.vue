<script setup lang="ts">
import { computed, ref } from 'vue'

export interface DirectoryApp {
  name: string
  title: string
  description: string
  path: string
}

const props = withDefaults(defineProps<{
  title: string
  app?: string
  apps?: DirectoryApp[]
  home?: string
}>(), { apps: () => [], home: '/' })

const drawer = ref<HTMLDialogElement | null>(null)
const open = ref(false)
const query = ref('')
const matches = computed(() => {
  const term = query.value.trim().toLocaleLowerCase()
  return props.apps.filter((app) => `${app.title} ${app.description}`.toLocaleLowerCase().includes(term))
    .sort((a, b) => a.title.localeCompare(b.title))
})

function showApps() {
  query.value = ''
  drawer.value?.showModal()
  open.value = true
}
</script>

<template>
  <header class="marina-bar">
    <div class="marina-title-row">
      <button class="marina-menu-button" type="button" aria-label="Open apps" aria-haspopup="dialog"
        :aria-expanded="open" @click="showApps">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">
          <path d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <a class="marina-brand" :href="home">Marina</a>
      <span class="marina-divider" aria-hidden="true">/</span>
      <span class="marina-app-title" :title="title">{{ title }}</span>
      <div class="marina-actions"><slot name="actions" /></div>
    </div>
    <nav v-if="$slots.nav" class="marina-app-nav" aria-label="This app"><slot name="nav" /></nav>
  </header>
  <dialog ref="drawer" class="marina-app-drawer" aria-labelledby="marina-apps-title" @close="open = false"
    @keydown.esc.prevent="drawer?.close()"
    @click="($event.target === drawer) && drawer?.close()">
    <div class="marina-drawer-content">
      <div class="marina-drawer-heading">
        <div><span class="marina-eyebrow">Marina</span><h2 id="marina-apps-title">Your apps</h2></div>
        <button type="button" class="marina-close-button" aria-label="Close apps" @click="drawer?.close()">×</button>
      </div>
      <template v-if="apps.length">
        <label class="marina-search-label" for="marina-app-search">Find an app</label>
        <input id="marina-app-search" v-model="query" type="search" placeholder="Search names and descriptions" autofocus />
        <p class="marina-result-count" role="status">{{ matches.length }} {{ matches.length === 1 ? 'app' : 'apps' }}</p>
        <nav class="marina-app-list" aria-label="Apps">
          <a v-for="item in matches" :key="item.name" :href="item.path"
            :aria-current="item.name === app ? 'page' : undefined">
            <span class="marina-item-title">{{ item.title }}<span v-if="item.name === app" class="marina-current">Current</span></span>
            <span class="marina-description">{{ item.description }}</span>
          </a>
        </nav>
        <p v-if="!matches.length" class="marina-empty">No matching apps. Try another name or topic.</p>
      </template>
      <p v-else class="marina-empty">Explore the Marina directory to find another app.</p>
      <a class="marina-directory-link" :href="home">All apps <span aria-hidden="true">↗</span></a>
    </div>
  </dialog>
</template>

<style scoped>
.marina-bar, .marina-app-drawer {
  color: var(--ink, #17181c);
  background: var(--panel, #fff);
  font: 15px/1.5 ui-sans-serif, system-ui, sans-serif;
}
.marina-bar { border-bottom: 1px solid var(--edge, #e2ded2); }
.marina-title-row { display: flex; align-items: center; gap: .75rem; padding: .5rem 1.25rem; min-width: 0; }
.marina-menu-button, .marina-close-button {
  display: grid; place-items: center; flex: none; width: 44px; height: 44px;
  padding: 0; color: inherit; background: transparent; border: 1px solid var(--edge, #e2ded2);
  border-radius: .5rem; cursor: pointer;
}
.marina-brand { color: inherit; text-decoration: none; font-weight: 650; }
.marina-divider { color: var(--muted, #6d6a60); }
.marina-app-title { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.marina-actions { margin-left: auto; display: flex; align-items: center; gap: .75rem; flex: none; }
.marina-app-nav { display: flex; gap: 1.25rem; overflow-x: auto; padding: 0 1.25rem; }
.marina-app-nav :slotted(a) { display: block; padding: .6rem 0; color: var(--muted, #6d6a60); text-decoration: none; white-space: nowrap; }
.marina-app-nav :slotted(a[aria-current='page']) { color: var(--mark, #1f5c4a); box-shadow: inset 0 -2px var(--mark, #1f5c4a); }
.marina-app-drawer {
  box-sizing: border-box; position: fixed; inset: 0 auto 0 0; margin: 0;
  width: min(28rem, 100%); height: 100%; height: 100dvh; max-height: none; max-width: 100%;
  padding: 0; border: 0; border-right: 1px solid var(--edge, #e2ded2);
}
.marina-app-drawer::backdrop { background: rgb(0 0 0 / .35); }
.marina-drawer-content { padding: 1.5rem; }
.marina-drawer-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.75rem; }
.marina-eyebrow { font-size: .75rem; letter-spacing: .1em; text-transform: uppercase; color: var(--muted, #6d6a60); }
h2 { margin: .2rem 0 0; font-size: 1.6rem; letter-spacing: -.03em; }
.marina-close-button { font-size: 1.6rem; }
.marina-search-label { display: block; font-size: .85rem; font-weight: 600; margin-bottom: .4rem; }
input { box-sizing: border-box; width: 100%; padding: .7rem; border: 1px solid var(--edge, #e2ded2); border-radius: .5rem; background: inherit; color: inherit; font: inherit; font-size: 16px; }
.marina-result-count, .marina-description, .marina-empty { color: var(--muted, #6d6a60); font-size: .875rem; }
.marina-app-list { display: grid; gap: .4rem; }
.marina-app-list a { display: grid; gap: .3rem; padding: .85rem; border: 1px solid transparent; border-radius: .6rem; color: inherit; text-decoration: none; overflow-wrap: anywhere; }
.marina-app-list a:hover, .marina-app-list a[aria-current='page'] { background: var(--paper, #fcfbf8); border-color: var(--edge, #e2ded2); }
.marina-item-title { display: flex; gap: .75rem; align-items: baseline; justify-content: space-between; font-weight: 650; }
.marina-current { font-size: .7rem; font-weight: 500; color: var(--mark, #1f5c4a); }
.marina-directory-link { display: block; margin-top: 1.5rem; padding: .75rem 0; border-top: 1px solid var(--edge, #e2ded2); color: var(--mark, #1f5c4a); }
:is(a, button, input):focus-visible { outline: 2px solid var(--mark, #1f5c4a); outline-offset: 3px; }
@media (max-width: 40rem) {
  .marina-title-row { padding: .5rem .75rem; gap: .5rem; }
  .marina-app-nav { padding: 0 .75rem; }
  .marina-brand, .marina-divider { display: none; }
}
</style>
