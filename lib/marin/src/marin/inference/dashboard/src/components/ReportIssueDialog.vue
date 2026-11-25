<script setup lang="ts">
import { ref } from 'vue'

import type { ReportInput } from '../lib/api'

const props = defineProps<{
  /** Save the report and return its storage URL. */
  save: (input: ReportInput) => Promise<string>
}>()

const FILER_URL = 'https://filer.marina.oa.dev/#'
const REPORTER_KEY = 'marin-serve-reporter'

const dialog = ref<HTMLDialogElement | null>(null)
const description = ref('')
const reporter = ref('')
const includeConversation = ref(true)
const status = ref<'editing' | 'saving' | 'saved' | 'failed'>('editing')
const location = ref('')

function open() {
  description.value = ''
  reporter.value = localStorage.getItem(REPORTER_KEY) ?? ''
  includeConversation.value = true
  status.value = 'editing'
  dialog.value?.showModal()
}

async function submit() {
  status.value = 'saving'
  const name = reporter.value.trim()
  localStorage.setItem(REPORTER_KEY, name)
  try {
    location.value = await props.save({
      description: description.value.trim(),
      reporter: name,
      includeConversation: includeConversation.value,
    })
    status.value = 'saved'
  } catch (error) {
    console.error('Could not save the feedback report', error)
    status.value = 'failed'
  }
}

function onBackdropClick(event: MouseEvent) {
  if (event.target === dialog.value && status.value !== 'saving') dialog.value?.close()
}

defineExpose({ open })
</script>

<template>
  <dialog
    ref="dialog"
    aria-labelledby="report-issue-title"
    class="m-auto w-[min(32rem,calc(100vw-2rem))] rounded-xl border border-surface-border bg-surface-raised p-0 text-text shadow-xl backdrop:bg-black/50"
    @click="onBackdropClick"
  >
    <div v-if="status === 'saved'" class="space-y-4 p-5">
      <h2 id="report-issue-title" class="text-sm font-semibold">Report saved</h2>
      <p class="text-sm leading-relaxed text-text-secondary">Thanks. The report is stored privately in Marin storage:</p>
      <p class="break-all rounded-lg bg-surface px-3 py-2 font-mono text-xs text-text-secondary">{{ location }}</p>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="rounded-lg px-3 py-1.5 text-sm text-text-muted transition-colors hover:text-text"
          @click="dialog?.close()"
        >
          Close
        </button>
        <a
          autofocus
          class="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover"
          :href="FILER_URL + encodeURIComponent(location)"
          target="_blank"
          rel="noopener"
        >
          Open in Filer
        </a>
      </div>
    </div>
    <form v-else class="space-y-4 p-5" @submit.prevent="submit">
      <h2 id="report-issue-title" class="text-sm font-semibold">Report a poor response</h2>
      <label class="block space-y-1.5 text-sm text-text-secondary">
        <span>What went wrong, and what would a good response look like?</span>
        <textarea
          v-model="description"
          required
          autofocus
          rows="4"
          class="w-full rounded-lg border border-surface-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
      </label>
      <label class="block space-y-1.5 text-sm text-text-secondary">
        <span>Your name <span class="text-xs text-text-muted">(optional, remembered in this browser)</span></span>
        <input
          v-model="reporter"
          type="text"
          autocomplete="name"
          class="w-full rounded-lg border border-surface-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
      </label>
      <label class="flex cursor-pointer items-start gap-2 text-sm text-text-secondary">
        <input v-model="includeConversation" type="checkbox" class="mt-0.5 accent-accent" />
        <span>
          Include the conversation
          <span class="block text-xs text-text-muted">
            Uncheck to send only your description, name, and the model and serving settings.
          </span>
        </span>
      </label>
      <p v-if="status === 'failed'" class="text-sm text-status-danger">
        The report could not be saved. Try again, or check the browser console.
      </p>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="rounded-lg px-3 py-1.5 text-sm text-text-muted transition-colors hover:text-text"
          :disabled="status === 'saving'"
          @click="dialog?.close()"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover disabled:opacity-50"
          :disabled="status === 'saving' || !description.trim()"
        >
          {{ status === 'saving' ? 'Saving…' : 'Save report' }}
        </button>
      </div>
    </form>
  </dialog>
</template>
