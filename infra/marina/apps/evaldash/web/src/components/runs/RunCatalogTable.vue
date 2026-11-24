<script setup lang="ts">
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import StatusChip from '@/components/shared/StatusChip.vue'
import type { RunRow } from '@/types/api'
import { formatTimestamp } from '@/utils/formatting'

const JOB_ROLES = ['serve', 'eval', 'orchestrator']

defineProps<{
  rows: RunRow[]
  group: string
  launchLocation: (group: string) => RouteLocationRaw
}>()

function jobLinks(row: RunRow) {
  return JOB_ROLES
    .filter((role) => row.jobs?.[role])
    .map((role) => ({
      role,
      path: row.jobs[role],
      url: `https://iris.oa.dev/#/job/${encodeURIComponent(row.jobs[role])}`,
    }))
}
</script>

<template>
  <table class="catalog-table" role="table" aria-label="Evaluation runs">
    <thead role="rowgroup">
      <tr role="row">
        <th scope="col" role="columnheader">Evaluation / run</th>
        <th class="environment-column" scope="col" role="columnheader">Owner / compute</th>
        <th class="date-column" scope="col" role="columnheader" aria-sort="descending">Created <span aria-hidden="true">↓</span></th>
        <th class="status-column" scope="col" role="columnheader">Status</th>
        <th class="action-column" scope="col" role="columnheader">Links</th>
      </tr>
    </thead>
    <tbody role="rowgroup">
      <tr v-for="row in rows" :key="row.run_id" role="row">
        <th class="identity-cell" scope="row" role="rowheader">
          <RouterLink class="record-title record-link" :to="{ name: 'run', params: { runId: row.run_id } }">
            {{ row.eval_name || row.run_id }}
          </RouterLink>
          <div v-if="row.model_name || row.version" class="record-model">
            <ModelName v-if="row.model_name" :model="row.model_name" />
            <span v-if="row.version" class="version-label font-mono">{{ row.version }}</span>
          </div>
          <div v-if="row.eval_name" class="record-id font-mono">{{ row.run_id }}</div>
          <p v-if="row.tasks?.length && (row.tasks.length > 1 || row.tasks[0] !== row.eval_name)" class="record-note" :title="row.tasks.join(', ')">
            {{ row.tasks.slice(0, 3).join(', ') }}<span v-if="row.tasks.length > 3"> +{{ row.tasks.length - 3 }} more</span>
          </p>
        </th>
        <td role="cell">
          <span class="mobile-label" aria-hidden="true">Owner / compute</span>
          <div>{{ row.user_name || '—' }}</div>
          <div v-if="row.accelerator" class="compute-label font-mono">{{ row.accelerator }}</div>
        </td>
        <td role="cell">
          <span class="mobile-label" aria-hidden="true">Created</span>
          <time :datetime="row.created_at">{{ formatTimestamp(row.created_at) }}</time>
        </td>
        <td role="cell">
          <span class="mobile-label" aria-hidden="true">Status</span>
          <StatusChip :status="row.status" />
        </td>
        <td role="cell">
          <span class="mobile-label" aria-hidden="true">Links</span>
          <RouterLink
            v-if="row.status === 'succeeded'"
            :to="{ name: 'samples', params: { runId: row.run_id } }"
            class="record-link samples-link"
            :aria-label="`Samples for ${row.eval_name || row.run_id}`"
          >Samples</RouterLink>
          <div v-if="JOB_ROLES.some((role) => row.jobs?.[role])" class="job-links">
            <a
              v-for="job in jobLinks(row)"
              :key="job.role"
              class="record-link"
              :href="job.url"
              :title="job.path"
              target="_blank"
              rel="noopener"
            >{{ job.role }} <span aria-hidden="true">↗</span></a>
          </div>
          <RouterLink
            v-if="!group && row.group_id && row.group_id !== row.run_id"
            :to="launchLocation(row.group_id)"
            class="record-link launch-shortcut"
          >View launch</RouterLink>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.catalog-table {
  width: 100%;
  table-layout: fixed;
  color: var(--c-text);
  font-size: 0.8125rem;
}

thead {
  background: var(--c-surface-raised);
}

.catalog-table th,
.catalog-table td {
  padding: 0.625rem 1rem;
  text-align: start;
  vertical-align: top;
  overflow-wrap: anywhere;
}

.catalog-table thead th {
  padding-block: 0.55rem;
  color: var(--c-text-secondary);
  font-size: 0.72rem;
  font-weight: 650;
}

.environment-column { width: 16%; }
.date-column { width: 16%; }
.status-column { width: 13%; }
.action-column { width: 15%; }

tbody tr + tr {
  border-top: 1px solid var(--c-surface-border-subtle);
}

tbody tr:hover,
tbody tr:focus-within {
  background: color-mix(in srgb, var(--c-accent) 5%, var(--c-surface));
}

tbody td {
  color: var(--c-text-secondary);
}

.identity-cell {
  font-weight: 400;
}

.record-title {
  color: var(--c-text);
  font-size: 0.9rem;
  font-weight: 650;
}

.record-link {
  color: var(--c-accent);
}

.record-link:hover,
.record-link:focus-visible {
  text-decoration: underline;
  text-underline-offset: 0.18em;
}

.record-model {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.2rem 0.55rem;
}

.record-model,
.record-note,
.record-id,
.compute-label {
  color: var(--c-text-secondary);
}

.record-model,
.record-note,
.record-id {
  margin-top: 0.26rem;
}

.version-label {
  max-width: 100%;
  padding: 0.04rem 0.33rem;
  border-radius: 0.22rem;
  background: var(--c-surface-sunken);
  color: var(--c-text-secondary);
  font-size: 0.7rem;
  font-weight: 500;
}

.record-note {
  font-size: 0.8rem;
  line-height: 1.45;
}

.record-id,
.compute-label {
  font-size: 0.73rem;
}

.compute-label {
  margin-top: 0.2rem;
}

.samples-link {
  display: inline-block;
  margin-bottom: 0.16rem;
  font-weight: 650;
}

.job-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.125rem 0.6rem;
}

.launch-shortcut {
  display: inline-block;
  margin-top: 0.2rem;
}

.mobile-label {
  display: none;
}

@media (max-width: 790px) {
  .catalog-table,
  .catalog-table tbody {
    display: block;
  }

  .catalog-table thead {
    position: absolute;
    width: 1px;
    height: 1px;
    clip-path: inset(50%);
    overflow: hidden;
    white-space: nowrap;
  }

  .catalog-table tbody tr {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.7rem 1rem;
    padding: 0.85rem 1rem;
  }

  .catalog-table tbody th,
  .catalog-table tbody td {
    padding: 0;
  }

  .identity-cell {
    grid-column: 1 / -1;
  }

  .mobile-label {
    display: block;
    margin-bottom: 0.14rem;
    color: var(--c-text-secondary);
    font-size: 0.7rem;
    font-weight: 600;
  }
}

@media (min-width: 490px) and (max-width: 790px) {
  .catalog-table tbody tr {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
}
</style>
