<script setup lang="ts">
import { computed, onUnmounted, ref } from 'vue'
import { importRepository, isAbortError } from '../lib/api'
import type { ShellWorkspace } from '../lib/types'

const props = defineProps<{
  workspace: ShellWorkspace | null
  disabled: boolean
}>()

const emit = defineEmits<{
  'update:workspace': [workspace: ShellWorkspace | null]
  close: []
  persist: []
}>()

const importing = ref(false)
const status = ref('')
let abort: AbortController | null = null

const repositoryUrl = computed({
  get: () => props.workspace?.repositoryUrl ?? '',
  set: (repositoryUrl: string) => {
    if (!props.workspace) return
    update({ ...props.workspace, repositoryUrl })
  },
})

const filesJson = computed({
  get: () => props.workspace?.filesJson ?? '',
  set: (value: string) => {
    if (!props.workspace) return
    update({ ...props.workspace, filesJson: value, history: [] })
    status.value = 'Command history reset because the initial files changed.'
  },
})

onUnmounted(stopImport)

function update(workspace: ShellWorkspace | null) {
  emit('update:workspace', workspace)
  emit('persist')
}

function enable() {
  status.value = ''
  update({ filesJson: '{}', history: [], repositoryUrl: '' })
}

function remove() {
  stopImport()
  status.value = ''
  update(null)
  emit('close')
}

function resetHistory() {
  if (!props.workspace) return
  update({ ...props.workspace, history: [] })
  status.value = 'Command history reset.'
}

function stopImport() {
  abort?.abort()
  abort = null
  importing.value = false
}

async function loadRepository() {
  const workspace = props.workspace
  if (!workspace || !workspace.repositoryUrl.trim() || importing.value || props.disabled) return
  importing.value = true
  status.value = ''
  const controller = new AbortController()
  abort = controller
  try {
    const snapshot = await importRepository(workspace.repositoryUrl.trim(), controller.signal)
    update({ ...workspace, filesJson: JSON.stringify(snapshot.files, null, 2), history: [] })
    status.value = `Loaded ${Object.keys(snapshot.files).length} text files; skipped ${snapshot.skipped_files}.`
  } catch (error) {
    if (!isAbortError(error)) status.value = String(error)
  } finally {
    if (abort === controller) stopImport()
  }
}
</script>

<template>
  <div class="mb-3 rounded-xl border border-surface-border bg-surface-raised p-3">
    <div v-if="!workspace" class="flex items-center justify-between gap-4">
      <p class="text-xs text-text-muted">
        Give the model an isolated filesystem, shell, Python, and simulated Git repository.
      </p>
      <button
        class="shrink-0 rounded-lg border border-accent px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/10"
        :disabled="disabled"
        @click="enable"
      >
        Enable workspace
      </button>
    </div>
    <template v-else>
      <div class="flex gap-2">
        <input
          v-model="repositoryUrl"
          type="url"
          :disabled="disabled || importing"
          placeholder="https://github.com/owner/repository"
          class="min-w-0 flex-1 rounded-lg border border-surface-border bg-surface-sunken px-3 py-1.5 font-mono text-xs text-text outline-none transition-colors focus:border-accent disabled:opacity-60"
          @keydown.enter.prevent="loadRepository"
        />
        <button
          class="rounded-lg border border-surface-border px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-text disabled:opacity-40"
          :disabled="disabled || importing || !repositoryUrl.trim()"
          @click="loadRepository"
        >
          {{ importing ? 'Loading…' : 'Load public repo' }}
        </button>
      </div>
      <p class="mt-1 text-[0.7rem] text-text-muted">
        Loads a bounded snapshot of a public GitHub repository's default branch. Remote history and .git are excluded.
      </p>
      <textarea
        v-model="filesJson"
        rows="7"
        :disabled="disabled || importing"
        spellcheck="false"
        placeholder="{&#10;  &quot;README.md&quot;: &quot;# Example repository\\n&quot;&#10;}"
        class="mt-3 w-full resize-y rounded-xl border border-surface-border bg-surface-sunken px-3 py-2 font-mono text-xs leading-relaxed text-text outline-none transition-colors focus:border-accent disabled:opacity-60"
      ></textarea>
      <div class="mt-2 flex flex-wrap items-center justify-between gap-2">
        <p class="text-[0.7rem] text-text-muted">
          Commands run from /work. {{ workspace.history.length }} commands will be replayed before the next call.
        </p>
        <div class="flex gap-2">
          <button
            class="text-xs text-text-muted transition-colors hover:text-text"
            :disabled="disabled || !workspace.history.length"
            @click="resetHistory"
          >
            Reset commands
          </button>
          <button
            class="text-xs text-status-danger transition-opacity hover:opacity-80"
            :disabled="disabled"
            @click="remove"
          >
            Remove workspace
          </button>
        </div>
      </div>
      <p v-if="status" class="mt-2 break-words text-[0.7rem] text-text-secondary">
        {{ status }}
      </p>
    </template>
  </div>
</template>
