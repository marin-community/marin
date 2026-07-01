<script setup lang="ts">
import { onMounted, ref } from 'vue'
import Sidebar from './components/Sidebar.vue'
import RunHeader from './components/RunHeader.vue'
import Tabs from './components/Tabs.vue'
import SummaryTab from './components/SummaryTab.vue'
import ChartsTab from './components/ChartsTab.vue'
import ProfileTab from './components/ProfileTab.vue'
import { useRun, type RunRef } from './composables/useRun'
import type { TabId } from './types'

const { manifest, config, summary, loading, error, updatedAt, load, refetch } = useRun()
const activeTab = ref<TabId>('summary')
const runRef = ref<RunRef | null>(null)
const sidebarOpen = ref(true)
const initial = ref<{ entity?: string; project?: string; user?: string }>({})

function syncUrl(r: RunRef, user?: string) {
  const params: Record<string, string> = { entity: r.entity, project: r.project, run: r.run_id }
  if (user) params.user = user
  history.replaceState(null, '', '?' + new URLSearchParams(params).toString())
}

function select(entity: string, project: string, name: string, user?: string) {
  activeTab.value = 'summary'
  runRef.value = { entity, project, run_id: name }
  syncUrl(runRef.value, user)
  load(runRef.value)
}

// Deep-link: an ?entity&project&run URL opens straight to that run and prefills the picker.
onMounted(() => {
  const p = new URLSearchParams(location.search)
  const entity = p.get('entity') ?? undefined
  const project = p.get('project') ?? undefined
  const user = p.get('user') ?? undefined
  initial.value = { entity, project, user }
  const run = p.get('run')
  if (entity && project && run) select(entity, project, run, user)
})
</script>

<template>
  <div class="flex h-screen">
    <Sidebar
      v-show="sidebarOpen"
      :initial-entity="initial.entity"
      :initial-project="initial.project"
      :initial-user="initial.user"
      @select="select"
      @collapse="sidebarOpen = false"
    />
    <main class="relative flex-1 overflow-auto">
      <button
        v-if="!sidebarOpen"
        class="m-3 rounded border border-surface-border bg-surface-raised px-3 py-1 text-sm text-accent hover:bg-accent-subtle"
        title="show sidebar"
        @click="sidebarOpen = true"
      >
        ☰ runs
      </button>

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
        <Tabs
          :active="activeTab"
          :has-profile="!!manifest.profile"
          :live="manifest.state === 'running' ? updatedAt : null"
          @change="(t) => (activeTab = t)"
        />
        <div class="p-6">
          <SummaryTab v-if="activeTab === 'summary'" :summary="summary" :config="config" />
          <ChartsTab
            v-else-if="activeTab === 'charts' && runRef"
            :run-ref="runRef"
            :columns="manifest.history.columns"
            :last-step="manifest.history.last_step"
          />
          <ProfileTab v-else-if="activeTab === 'profile' && runRef" :run-ref="runRef" />
          <p v-else class="text-text-muted">no profile on this run</p>
        </div>
      </div>

      <p v-else class="p-6 text-text-muted">pick an entity / project, then a run from the list</p>
    </main>
  </div>
</template>
