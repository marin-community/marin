<script setup lang="ts">
/**
 * On-demand LLM review of a sample set: asks the server to sample the current task+filter and
 * summarize the failure modes. Never auto-runs — the call is behind a button and degrades to a
 * plain "not configured" note when the reviewer has no API key.
 */
import { ref, watch } from 'vue'
import { apiPost } from '@/composables/useApi'
import type { ReviewResponse } from '@/types/api'
import type { SampleFilter } from '@/utils/samples'

const props = defineProps<{ runId: string; task: string; filter: SampleFilter }>()

const result = ref<ReviewResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

// A new task/filter invalidates a prior review — it summarized a different set.
watch(
  () => [props.task, props.filter],
  () => {
    result.value = null
    error.value = null
  },
)

async function run() {
  loading.value = true
  error.value = null
  try {
    result.value = await apiPost<ReviewResponse>(`api/runs/${props.runId}/samples/review`, {
      task: props.task,
      filter: props.filter,
      n: 20,
    })
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="flex items-center gap-3 flex-wrap">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Failure review</h3>
      <span class="text-xs text-text-muted">
        summarize the failure modes across the
        <span class="font-mono">{{ filter === 'all' ? '' : filter + ' ' }}{{ task }}</span> samples
      </span>
      <button
        class="ml-auto px-3 py-1.5 rounded-lg bg-accent text-surface text-sm font-medium hover:bg-accent-hover disabled:opacity-50"
        :disabled="loading"
        @click="run"
      >{{ loading ? 'Reviewing…' : result ? 'Re-run review' : 'Review with an LLM →' }}</button>
    </div>

    <p v-if="error" class="mt-3 text-sm text-status-danger">{{ error }}</p>

    <p v-else-if="result && !result.available" class="mt-3 text-sm text-text-muted">
      Reviewer unavailable — {{ result.reason }}. Set <span class="font-mono">ANTHROPIC_API_KEY</span> on the service to enable it.
    </p>

    <div v-else-if="result && result.summary" class="mt-3 space-y-3">
      <div class="text-xs text-text-muted font-mono">
        {{ result.model }} · reviewed {{ result.n_reviewed }} sample{{ result.n_reviewed === 1 ? '' : 's' }}
      </div>
      <p class="text-sm leading-relaxed">{{ result.summary.narrative }}</p>
      <div class="space-y-1.5">
        <div
          v-for="cat in result.summary.categories"
          :key="cat.label"
          class="flex items-baseline gap-3 rounded border border-surface-border-subtle px-3 py-2"
        >
          <span class="font-mono text-sm tabular-nums text-accent w-8">{{ cat.count }}</span>
          <span class="text-sm">{{ cat.label }}</span>
          <span v-if="cat.doc_ids.length" class="ml-auto font-mono text-[11px] text-text-muted">
            docs {{ cat.doc_ids.join(', ') }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
