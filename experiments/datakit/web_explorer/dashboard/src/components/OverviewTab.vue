<script setup lang="ts">
import { computed } from 'vue'
import type { Overview } from '../types'
import { fmt } from '../utils/formatting'
import StatRow from './StatRow.vue'

const props = defineProps<{ ov: Overview }>()

const stats = computed(() => {
  const counters = props.ov.counters || {}
  return [
    { value: fmt(props.ov.sources.length), label: 'sources' },
    { value: fmt(props.ov.buckets.length), label: `buckets (k=${props.ov.cluster_view})` },
    { value: fmt(counters['datakit_store/records_in']), label: 'records in' },
    { value: fmt(counters['datakit_store/records_out']), label: 'records out (stored)' },
    { value: fmt(counters['datakit_store/contaminated_dropped']), label: 'decontam dropped' },
    { value: fmt(counters['datakit_store/dedup_noncanonical_dropped']), label: 'dedup dropped' },
  ]
})

function coverage(stage: keyof Overview['resolved']): string {
  return `${(props.ov.resolved[stage] || []).length}/${props.ov.sources.length}`
}
</script>

<template>
  <div class="mb-3.5 rounded-lg border border-surface-border bg-surface p-4">
    <StatRow :stats="stats" />
  </div>
  <div class="rounded-lg border border-surface-border bg-surface p-4">
    <h3 class="mb-2 font-semibold">Stage coverage (resolved via lineage)</h3>
    <p class="mb-2 text-sm text-text-muted">
      data prefix <code class="rounded bg-surface-sunken px-1 font-mono">{{ ov.data_prefix }}</code> · tokenizer
      <code class="rounded bg-surface-sunken px-1 font-mono">{{ ov.tokenizer }}</code>
    </p>
    <table class="w-full border-collapse text-[13px]">
      <tbody>
        <tr class="border-b border-surface-border">
          <td class="px-2 py-1">normalize</td>
          <td class="px-2 py-1">{{ coverage('normalize') }}</td>
        </tr>
        <tr class="border-b border-surface-border">
          <td class="px-2 py-1">decontam</td>
          <td class="px-2 py-1">{{ coverage('decontam') }}</td>
        </tr>
        <tr class="border-b border-surface-border">
          <td class="px-2 py-1">cluster_assign</td>
          <td class="px-2 py-1">{{ coverage('cluster_assign') }}</td>
        </tr>
        <tr class="border-b border-surface-border">
          <td class="px-2 py-1">quality</td>
          <td class="px-2 py-1">
            {{ coverage('quality') }}
            <span v-if="!ov.resolved.quality.length" class="text-text-muted">
              (ambiguous — set WEB_EXPLORER_QUALITY_MODEL + WEB_EXPLORER_DOMAIN_CENTROIDS)
            </span>
          </td>
        </tr>
        <tr class="border-b border-surface-border">
          <td class="px-2 py-1">dedup</td>
          <td class="px-2 py-1">
            <template v-if="ov.dedup">{{ ov.dedup }}</template>
            <span v-else class="text-text-muted">unresolved</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>
