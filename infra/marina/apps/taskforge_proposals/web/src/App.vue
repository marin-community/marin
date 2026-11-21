<script setup>
import { computed, onMounted, onUnmounted, ref } from "vue";

import BrowseView from "./components/BrowseView.vue";
import CapabilitiesView from "./components/CapabilitiesView.vue";
import CapabilityView from "./components/CapabilityView.vue";
import ProposalView from "./components/ProposalView.vue";
import StatsView from "./components/StatsView.vue";
import { api, compact, filtersOf, href, num, parseRoute, ui } from "./lib.js";

const route = ref(parseRoute());
const meta = ref(null);
const metaError = ref("");
const onHash = () => { route.value = parseRoute(); window.scrollTo(0, 0); };

onMounted(async () => {
  window.addEventListener("hashchange", onHash);
  try { meta.value = await api("meta"); } catch (error) { metaError.value = error.message; }
});
onUnmounted(() => window.removeEventListener("hashchange", onHash));

const ingest = computed(() => meta.value?.stats?.ingest);
const shared = computed(() => filtersOf(route.value.query));

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("tf-theme", root.dataset.theme); } catch { /* storage unavailable */ }
}
</script>

<template>
  <header class="topbar">
    <a class="brand" :href="href('/proposals')">Taskforge <span>proposals · n=25</span></a>
    <nav class="topnav">
      <a :class="{ on: route.name === 'proposals' || route.name === 'p' }" :href="href('/proposals', shared)">Proposals</a>
      <a :class="{ on: route.name === 'capabilities' || route.name === 'c' }" :href="href('/capabilities')">Capabilities</a>
      <a :class="{ on: route.name === 'stats' }" :href="href('/stats', shared)">Stats</a>
    </nav>
    <div class="spacer"></div>
    <span v-if="meta" class="muted small nowrap" :title="ingest ? `Loaded ${ingest.finished_at}` : ''">
      {{ num(meta.counts.proposals) }} slots · {{ num(meta.counts.capabilities) }} capabilities
      <template v-if="ingest"> · loaded {{ ingest.finished_at.slice(0, 16).replace('T', ' ') }} UTC</template>
    </span>
    <div class="topbar-actions">
      <button class="ghost" title="Toggle light and dark" @click="toggleTheme">Theme</button>
    </div>
  </header>
  <main>
    <div v-if="metaError" class="error">Could not load the run summary: {{ metaError }}</div>
    <template v-else-if="meta">
      <ProposalView v-if="route.name === 'p'" :key="route.parts.join('/')" :capability-id="route.parts[0]" :slot-number="Number(route.parts[1])" />
      <CapabilityView v-else-if="route.name === 'c'" :key="route.parts[0]" :capability-id="route.parts[0]" />
      <CapabilitiesView v-else-if="route.name === 'capabilities'" :query="route.query" :meta="meta" />
      <StatsView v-else-if="route.name === 'stats'" :query="route.query" :meta="meta" />
      <BrowseView v-else :query="route.query" :meta="meta" />
    </template>
    <div v-else class="loading">Loading the run…</div>
  </main>
  <div v-if="ui.tip" class="tooltip" :style="{ left: `${ui.tip.x + 14}px`, top: `${ui.tip.y + 14}px` }">
    <div class="tt-title">{{ ui.tip.title }}</div>
    <div v-for="(row, i) in ui.tip.rows" :key="i" class="tt-row">{{ row }}</div>
  </div>
  <div v-if="ui.toast" class="toast">{{ ui.toast }}</div>
</template>
