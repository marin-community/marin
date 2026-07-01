<script setup lang="ts">
import { ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import RunHeader from './components/RunHeader.vue'
import Tabs from './components/Tabs.vue'
import SummaryTab from './components/SummaryTab.vue'
import { useRun } from './composables/useRun'
import type { TabId } from './types'

// Phase 2: run selection → mirror/poll/load → header + summary tab. Charts and
// profile land in phases 3–4 (see the migration plan under .agents/projects/).
const { manifest, config, summary, loading, error, load, refetch } = useRun()
const activeTab = ref<TabId>('summary')

function select(entity: string, project: string, name: string) {
  activeTab.value = 'summary'
  load({ entity, project, run_id: name })
}
</script>

<template>
  <div class="flex h-screen">
    <Sidebar @select="select" />
    <main class="relative flex-1 overflow-auto">
      <div v-if="loading" class="flex h-full items-center justify-center">
        <div class="text-center">
          <div class="mx-auto mb-4 h-14 w-14 animate-spin rounded-full border-[5px] border-surface-border border-t-accent"></div>
          <div class="text-lg font-semibold text-accent">{{ loading.verb }}</div>
          <div class="mt-1 text-sm text-text-muted">{{ loading.detail }}</div>
        </div>
      </div>

      <div v-else-if="error" class="p-6 text-status-danger">{{ error }}</div>

      <div v-else-if="manifest" class="flex flex-col">
        <div class="p-6 pb-3"><RunHeader :manifest="manifest" @refetch="refetch" /></div>
        <Tabs :active="activeTab" :has-profile="!!manifest.profile" @change="(t) => (activeTab = t)" />
        <div class="p-6">
          <SummaryTab v-if="activeTab === 'summary'" :summary="summary" :config="config" />
          <p v-else-if="activeTab === 'charts'" class="text-text-muted">charts — coming in phase 3</p>
          <p v-else class="text-text-muted">profile — coming in phase 4</p>
        </div>
      </div>

      <p v-else class="p-6 text-text-muted">pick an entity / project, then a run from the list</p>
    </main>
  </div>
</template>
