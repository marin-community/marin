<script setup lang="ts">
import { onUnmounted, ref, watch } from 'vue'
import { streamSse } from '../lib/api'
import { COMPLETION_EXAMPLES } from '../lib/examples'
import type { SamplingParams } from '../lib/types'

const props = defineProps<{
  params: SamplingParams
  model: string
}>()

const PROMPT_KEY = 'marin-quick-serve:completion-prompt:v1'

function storedPrompt(): string {
  try {
    return localStorage.getItem(PROMPT_KEY) ?? COMPLETION_EXAMPLES[0]
  } catch {
    return COMPLETION_EXAMPLES[0]
  }
}

const prompt = ref(storedPrompt())
const output = ref('')
const error = ref('')
const busy = ref(false)
const started = ref(false)
let abort: AbortController | null = null

watch(prompt, (value) => {
  try {
    localStorage.setItem(PROMPT_KEY, value)
  } catch {}
})

onUnmounted(() => abort?.abort())

function stopStreaming() {
  abort?.abort()
  abort = null
}

function onKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run()
}

async function run() {
  if (busy.value || !prompt.value.trim()) return
  busy.value = true
  started.value = true
  output.value = ''
  error.value = ''
  abort = new AbortController()
  try {
    await streamSse(
      'v1/completions',
      {
        model: props.model,
        prompt: prompt.value,
        stream: true,
        temperature: props.params.temperature,
        max_tokens: props.params.maxTokens,
        top_p: props.params.topP,
      },
      abort.signal,
      (data) => {
        const text = data.choices?.[0]?.text
        if (text) output.value += text
      },
    )
  } catch (err) {
    if (!(err instanceof DOMException && err.name === 'AbortError')) error.value = String(err)
  } finally {
    busy.value = false
    abort = null
  }
}
</script>

<template>
  <div class="min-h-0 flex-1 overflow-y-auto">
    <div class="mx-auto max-w-3xl space-y-4 px-4 py-5 md:px-6">
      <div>
        <label class="mb-1.5 block text-xs uppercase tracking-wide text-text-muted">Prompt</label>
        <textarea
          v-model="prompt"
          rows="4"
          class="w-full resize-y rounded-xl border border-surface-border bg-surface-raised px-3.5 py-2.5 font-mono text-sm leading-relaxed text-text outline-none transition-colors focus:border-accent"
          @keydown="onKeydown"
        ></textarea>
        <div class="mt-2 flex flex-wrap items-center gap-2">
          <button
            v-for="example in COMPLETION_EXAMPLES"
            :key="example"
            class="rounded-full border border-surface-border px-3 py-1 font-mono text-xs text-text-muted transition-colors hover:border-accent hover:text-text"
            @click="prompt = example"
          >
            {{ example }}
          </button>
          <div class="ml-auto flex items-center gap-2">
            <button
              v-if="busy"
              class="rounded-lg border border-surface-border px-3 py-1.5 text-sm text-text-secondary transition-colors hover:border-status-danger hover:text-status-danger"
              @click="stopStreaming"
            >
              Stop
            </button>
            <button
              class="rounded-lg bg-accent px-4 py-1.5 text-sm font-semibold text-surface transition-colors hover:bg-accent-hover disabled:opacity-40"
              :disabled="busy || !prompt.trim()"
              @click="run"
            >
              {{ busy ? 'Generating…' : 'Generate' }}
            </button>
          </div>
        </div>
        <div class="mt-1.5 text-xs text-text-muted">Cmd/Ctrl+Enter to run. The model continues the prompt.</div>
      </div>

      <div v-if="started || error">
        <label class="mb-1.5 block text-xs uppercase tracking-wide text-text-muted">Output</label>
        <div
          class="min-h-16 whitespace-pre-wrap break-words rounded-xl border border-surface-border bg-surface-sunken px-3.5 py-2.5 font-mono text-sm leading-relaxed"
        >
          <span v-if="error" class="text-status-danger">{{ error }}</span>
          <template v-else>
            <span class="text-text-muted">{{ prompt }}</span><span class="text-text">{{ output }}</span>
            <span v-if="busy" class="animate-pulse text-text-muted">▍</span>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
