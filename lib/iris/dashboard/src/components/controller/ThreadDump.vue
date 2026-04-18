<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { controllerRpcCall, workerRpcCall } from '@/composables/useRpc'
import PageShell from '@/components/layout/PageShell.vue'

const route = useRoute()
const router = useRouter()

// Derive the RPC target from route params/path.
// Backend target conventions:
//   /system/process              — the controller process itself
//   /system/worker/<worker_id>   — a worker process (NOT a task on the worker)
//   /<user>/<job>/<task_index>   — a user's task attempt
const taskId = computed(() => (route.params.taskId as string) ?? '')
const jobId = computed(() => (route.params.jobId as string) ?? '')
const workerId = computed(() => (route.params.workerId as string) ?? '')

const target = computed(() => {
  if (route.path.startsWith('/system/controller')) return '/system/process'
  if (workerId.value) return `/system/worker/${workerId.value}`
  return taskId.value
})

const displayTarget = computed(() => {
  if (route.path.startsWith('/system/controller')) return 'Controller'
  if (workerId.value) return `Worker: ${workerId.value}`
  return taskId.value
})

// Use controllerRpcCall in controller dashboard, workerRpcCall in worker dashboard.
// Worker dashboard routes set meta.rpc = 'worker'.
const rpcCall = computed(() => {
  return route.meta.rpc === 'worker' ? workerRpcCall : controllerRpcCall
})

const backTo = computed(() => {
  if (jobId.value) return `/job/${encodeURIComponent(jobId.value)}`
  if (workerId.value) return `/worker/${encodeURIComponent(workerId.value)}`
  return '/'
})

const backLabel = computed(() => {
  if (jobId.value) return 'Back'
  if (workerId.value) return 'Worker'
  if (route.path.startsWith('/system/controller')) return 'Dashboard'
  return 'Back'
})

const threadDump = ref('')
const loading = ref(false)
const error = ref<string | null>(null)
const lastFetched = ref<string | null>(null)
const includeLocals = ref(route.query.locals === '1')

watch(includeLocals, (val) => {
  router.replace({ query: { ...route.query, locals: val ? '1' : undefined } })
})

async function fetchThreadDump() {
  const t = target.value
  if (!t) {
    error.value = 'No target provided'
    return
  }
  loading.value = true
  error.value = null
  try {
    const body = {
      target: t,
      durationSeconds: 10,
      profileType: { threads: { locals: includeLocals.value } },
    }
    const resp = await rpcCall.value<{ profileData?: string; error?: string }>('ProfileTask', body)
    if (resp.error) {
      error.value = resp.error
      return
    }
    if (resp.profileData) {
      threadDump.value = atob(resp.profileData)
      lastFetched.value = new Date().toLocaleTimeString()
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(fetchThreadDump)
</script>

<template>
  <PageShell :title="`Thread Dump: ${displayTarget}`" :back-to="backTo" :back-label="backLabel">
    <!-- Controls -->
    <div class="flex items-center gap-3 -mt-4 mb-4">
      <button
        class="px-3 py-1.5 text-xs font-medium border border-surface-border rounded hover:bg-surface-raised text-text-secondary disabled:opacity-50"
        :disabled="loading"
        @click="fetchThreadDump"
      >
        {{ loading ? '⏳ Fetching...' : '↻ Refresh' }}
      </button>
      <label class="flex items-center gap-1.5 text-xs text-text-secondary cursor-pointer select-none">
        <input
          v-model="includeLocals"
          type="checkbox"
          class="rounded border-surface-border"
        />
        Include locals
      </label>
      <span v-if="lastFetched" class="text-xs text-text-muted">
        Last fetched: {{ lastFetched }}
      </span>
      <span class="text-xs text-text-muted ml-auto">
        Tip: Ctrl-R to refresh the page and re-fetch the thread dump
      </span>
    </div>

    <!-- Loading (first load) -->
    <div v-if="loading && !threadDump" class="flex items-center justify-center py-12 text-text-muted text-sm">
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      Fetching thread dump...
    </div>

    <!-- Error -->
    <div
      v-else-if="error"
      class="px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
    >
      {{ error }}
    </div>

    <!-- Thread dump content -->
    <div v-else-if="threadDump" class="rounded-lg border border-surface-border bg-surface-sunken overflow-auto">
      <pre class="p-4 text-xs font-mono text-text whitespace-pre-wrap break-words leading-relaxed">{{ threadDump }}</pre>
    </div>

    <!-- No data -->
    <div v-else class="text-sm text-text-muted py-8 text-center">
      No thread dump data
    </div>
  </PageShell>
</template>
