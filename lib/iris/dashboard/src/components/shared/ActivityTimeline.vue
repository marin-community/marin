<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { RESOURCE_MESSAGES, RESOURCE_TYPES, useListResources } from '@/composables/useResources'
import type { ResourceActivityEntry, ResourceKey } from '@/types/rpc'
import { formatRelativeTime, timestampMs } from '@/utils/formatting'
import EmptyState from '@/components/shared/EmptyState.vue'
import SourceWarnings from '@/components/shared/SourceWarnings.vue'

const props = defineProps<{ target: ResourceKey; attemptUid?: string }>()
const { data, loading, error, refresh } = useListResources<ResourceActivityEntry>(
  RESOURCE_TYPES.activityEntry,
  RESOURCE_MESSAGES.activityQuery,
  () => ({ target: props.target, attemptUid: props.attemptUid, page: { pageSize: 200 } }),
  'FULL',
)
const entries = computed(() => data.value?.items ?? [])
onMounted(refresh)
watch(() => props.attemptUid, refresh)
</script>

<template>
  <section>
    <h3 class="font-semibold mb-2">Activity</h3>
    <SourceWarnings :statuses="data?.page?.sourceStatuses" />
    <div v-if="error" class="text-sm text-status-danger">{{ error }}</div>
    <div v-else-if="loading && entries.length === 0" class="text-sm text-text-muted">Loading activity…</div>
    <EmptyState v-else-if="entries.length === 0" message="No activity recorded" />
    <ol v-else class="border rounded divide-y">
      <li v-for="entry in entries" :key="entry.entryId" class="p-3 flex gap-3 text-sm">
        <span class="w-28 shrink-0 text-xs text-text-muted">{{ formatRelativeTime(timestampMs(entry.occurredAt)) }}</span>
        <div class="min-w-0">
          <div><span class="font-semibold">{{ entry.kind }}</span> <span class="text-text-muted">· {{ entry.source }}</span></div>
          <div class="break-words">{{ entry.message }}</div>
        </div>
      </li>
    </ol>
  </section>
</template>
