<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{ submit: [includeConversation: boolean] }>()

const dialog = ref<HTMLDialogElement | null>(null)
const includeConversation = ref(true)

function open() {
  includeConversation.value = true
  dialog.value?.showModal()
}

function submit() {
  dialog.value?.close()
  // Emitted inside the click handler, so the caller can open a tab without a popup block.
  emit('submit', includeConversation.value)
}

function onBackdropClick(event: MouseEvent) {
  if (event.target === dialog.value) dialog.value?.close()
}

defineExpose({ open })
</script>

<template>
  <dialog
    ref="dialog"
    aria-labelledby="report-issue-title"
    class="m-auto w-[min(28rem,calc(100vw-2rem))] rounded-xl border border-surface-border bg-surface-raised p-0 text-text shadow-xl backdrop:bg-black/50"
    @click="onBackdropClick"
  >
    <form class="space-y-4 p-5" @submit.prevent="submit">
      <h2 id="report-issue-title" class="text-sm font-semibold">Report a poor response</h2>
      <p class="text-sm leading-relaxed text-text-secondary">
        This opens a prefilled issue in the public
        <span class="font-mono text-xs">marin-community/marin</span> repository. Nothing is posted until you
        submit it on GitHub. Review the draft and remove private or sensitive content first.
      </p>
      <label class="flex cursor-pointer items-start gap-2 text-sm text-text-secondary">
        <input v-model="includeConversation" type="checkbox" class="mt-0.5 accent-accent" />
        <span>
          Include the conversation
          <span class="block text-xs text-text-muted">
            Uncheck to report only the model and settings. The title, transcript, and custom template
            instructions are then omitted.
          </span>
        </span>
      </label>
      <div class="flex justify-end gap-2">
        <button
          type="button"
          class="rounded-lg px-3 py-1.5 text-sm text-text-muted transition-colors hover:text-text"
          @click="dialog?.close()"
        >
          Cancel
        </button>
        <button
          type="submit"
          autofocus
          class="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-surface transition-colors hover:bg-accent-hover"
        >
          Open GitHub draft
        </button>
      </div>
    </form>
  </dialog>
</template>
