<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { formatTimestamp } from '@/utils/formatting'
import type { ReportDetail } from '@/types/api'
import PlaygroundTranscript from '@/components/reports/PlaygroundTranscript.vue'

const props = defineProps<{ reportId: string }>()

const FILER_URL = 'https://filer.marina.oa.dev/#'
const SERVING_FIELDS = ['backend', 'dtype', 'tensor_parallel_size', 'max_model_len', 'endpoint'] as const

const { data, loading, error, refresh } = useApi<ReportDetail>(
  () => `api/playground-reports/${encodeURIComponent(props.reportId)}`,
)
onMounted(refresh)
watch(() => props.reportId, refresh)

const servingRows = computed(() =>
  SERVING_FIELDS.filter((key) => data.value?.serving[key] != null).map((key) => [key, String(data.value!.serving[key])]),
)

</script>

<template>
  <section class="space-y-4">
    <RouterLink to="/reports" class="text-sm text-accent hover:text-accent-hover hover:underline">← All reports</RouterLink>

    <div v-if="error" class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2">
      {{ error }}
    </div>
    <div v-if="loading && !data" class="text-sm text-text-muted py-12 text-center">Loading…</div>

    <template v-if="data">
      <div class="flex flex-wrap items-baseline gap-3">
        <h2 class="text-lg font-semibold font-mono">{{ data.report.model }}</h2>
        <span class="text-sm text-text-secondary">
          {{ formatTimestamp(data.created_at) }} · {{ data.report.reporter || 'anonymous' }}
        </span>
      </div>

      <div class="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <div class="space-y-4">
          <div class="rounded-lg border border-surface-border p-4">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">What went wrong</h3>
            <p class="text-sm whitespace-pre-wrap">{{ data.report.description }}</p>
          </div>

          <div class="rounded-lg border border-surface-border p-4">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
              Conversation<template v-if="data.report.conversation?.title"> · {{ data.report.conversation.title }}</template>
            </h3>
            <PlaygroundTranscript v-if="data.report.conversation" :conversation="data.report.conversation" />
            <p v-else class="text-sm text-text-muted">The reporter left the conversation out.</p>
          </div>
        </div>

        <div class="space-y-4">
          <div class="rounded-lg border border-surface-border p-4 space-y-2 text-sm">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Serving</h3>
            <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
              <template v-for="[key, value] in servingRows" :key="key">
                <dt class="text-text-muted">{{ key }}</dt>
                <dd class="font-mono text-[13px] break-all">{{ value }}</dd>
              </template>
              <template v-if="data.report.conversation?.thinkingMode">
                <dt class="text-text-muted">thinking</dt>
                <dd class="font-mono text-[13px]">{{ data.report.conversation.thinkingMode }}</dd>
              </template>
            </dl>
            <a
              :href="FILER_URL + encodeURIComponent(data.source_url)"
              target="_blank"
              rel="noopener"
              class="inline-block text-xs text-accent hover:text-accent-hover hover:underline break-all"
            >Raw report in Filer ↗</a>
          </div>
        </div>
      </div>
    </template>
  </section>
</template>
