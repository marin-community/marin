<script setup lang="ts">
/**
 * Group membership for a run: sibling runs sharing its group_id (one serve group evaluates N
 * evals against one model). Hidden for standalone runs (group_id == run_id and no siblings).
 */
import { computed, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useApi } from '@/composables/useApi'
import { formatTimestamp } from '@/utils/formatting'
import type { GroupResponse } from '@/types/api'
import StatusChip from '@/components/shared/StatusChip.vue'

const props = defineProps<{ runId: string }>()

const { data, refresh } = useApi<GroupResponse>(() => `api/runs/${props.runId}/group`)

onMounted(refresh)
watch(() => props.runId, refresh)

const siblings = computed(() => data.value?.siblings ?? [])
const show = computed(() => siblings.value.length > 0)
</script>

<template>
  <div v-if="show" class="rounded-lg border border-surface-border bg-surface p-4">
    <div class="flex items-center gap-2 flex-wrap mb-2">
      <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Serve group</h3>
      <code class="font-mono text-[12px] text-text-secondary break-all">{{ data?.group_id }}</code>
      <RouterLink
        :to="`/runs?group=${encodeURIComponent(data?.group_id ?? '')}`"
        class="text-xs text-accent hover:text-accent-hover hover:underline ml-auto"
      >view all in group →</RouterLink>
    </div>
    <ul class="rounded-md border border-surface-border-subtle overflow-hidden">
      <li v-for="s in siblings" :key="s.run_id">
        <RouterLink
          :to="`/runs/${s.run_id}`"
          class="flex items-center gap-2.5 px-3 py-2 text-sm border-b border-surface-border-subtle last:border-b-0 hover:bg-surface-raised"
        >
          <StatusChip :status="s.status" />
          <span class="font-mono text-[13px] text-accent">{{ s.eval_name ?? s.run_id }}</span>
          <span class="text-xs text-text-muted ml-auto">{{ formatTimestamp(s.created_at) }}</span>
        </RouterLink>
      </li>
    </ul>
  </div>
</template>
