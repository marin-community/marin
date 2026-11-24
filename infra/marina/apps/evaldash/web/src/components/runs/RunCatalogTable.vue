<script setup lang="ts">
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import StatusChip from '@/components/shared/StatusChip.vue'
import { RUN_STATUS, type RunRow } from '@/types/api'
import { formatTimestamp, irisJobUrl } from '@/utils/formatting'

defineProps<{
  rows: RunRow[]
  group: string
  launchLocation: (group: string) => RouteLocationRaw
}>()
</script>

<template>
  <table class="catalog-table" role="table" aria-label="Evaluation runs">
    <thead role="rowgroup">
      <tr role="row">
        <th scope="col" role="columnheader">Evaluation / model</th>
        <th class="environment-column" scope="col" role="columnheader">Owner / compute</th>
        <th class="date-column" scope="col" role="columnheader" aria-sort="descending">Created <span aria-hidden="true">↓</span></th>
        <th class="status-column" scope="col" role="columnheader">Status</th>
        <th class="action-column" scope="col" role="columnheader">Links</th>
      </tr>
    </thead>
    <tbody role="rowgroup">
      <tr v-for="row in rows" :key="row.run_id" role="row">
        <th class="identity-cell" scope="row" role="rowheader">
          <div class="record-identity">
            <RouterLink class="record-title record-link" :to="{ name: 'run', params: { runId: row.run_id } }" :title="row.run_id">
              {{ row.eval_name || row.run_id }}
            </RouterLink>
            <ModelName v-if="row.model_name" :model="row.model_name" class="record-model" />
          </div>
        </th>
        <td role="cell">
          <div class="record-environment">
            <span v-if="row.user_name">{{ row.user_name }}</span>
            <span v-if="row.accelerator">{{ row.accelerator }}</span>
          </div>
        </td>
        <td role="cell">
          <time :datetime="row.created_at">{{ formatTimestamp(row.created_at) }}</time>
        </td>
        <td role="cell">
          <StatusChip :status="row.status" />
        </td>
        <td role="cell">
          <div class="record-actions">
            <RouterLink
              v-if="row.status === RUN_STATUS.SUCCEEDED"
              :to="{ name: 'samples', params: { runId: row.run_id } }"
              class="record-link samples-link"
              :aria-label="`Samples for ${row.eval_name || row.run_id}`"
            >Samples</RouterLink>
            <details class="record-details">
              <summary class="record-link">Details</summary>
              <div class="details-content">
                <dl>
                  <dt>Run ID</dt>
                  <dd><code>{{ row.run_id }}</code></dd>
                  <template v-if="row.version">
                    <dt>Version</dt>
                    <dd>{{ row.version }}</dd>
                  </template>
                  <template v-if="row.tasks?.length && (row.tasks.length > 1 || row.tasks[0] !== row.eval_name)">
                    <dt>Tasks</dt>
                    <dd>{{ row.tasks.join(', ') }}</dd>
                  </template>
                </dl>
                <div class="detail-links">
                  <RouterLink
                    v-if="!group && row.group_id && row.group_id !== row.run_id"
                    :to="launchLocation(row.group_id)"
                    class="record-link"
                  >View launch</RouterLink>
                  <a
                    v-for="(path, role) in row.jobs"
                    :key="role"
                    class="record-link"
                    :href="irisJobUrl(path)"
                    :title="path"
                    target="_blank"
                    rel="noopener"
                  >{{ role }} job <span aria-hidden="true">↗</span></a>
                </div>
              </div>
            </details>
          </div>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.catalog-table {
  width: 100%;
  table-layout: fixed;
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
}

thead {
  background: var(--c-surface-raised);
}

.catalog-table th,
.catalog-table td {
  padding: 0.48rem 0.85rem;
  text-align: start;
  vertical-align: middle;
  overflow-wrap: anywhere;
}

.catalog-table thead th {
  padding-block: 0.42rem;
  font-size: 0.75rem;
  font-weight: 650;
}

.environment-column { width: 14%; }
.date-column { width: 18%; }
.status-column { width: 11%; }
.action-column { width: 13%; }

tbody tr + tr {
  border-top: 1px solid var(--c-surface-border-subtle);
}

tbody tr:hover,
tbody tr:focus-within {
  background: color-mix(in srgb, var(--c-accent) 5%, var(--c-surface));
}

.identity-cell {
  font-weight: 400;
}

.record-identity,
.record-environment,
.record-actions,
.detail-links {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.12rem 0.65rem;
}

.record-title {
  font-size: 0.875rem;
  font-weight: 650;
}

.record-model {
  min-width: 0;
}

.record-environment span + span::before {
  content: '·';
  display: inline-block;
  margin-inline-end: 0.65rem;
}

.record-link {
  color: var(--c-accent);
}

.record-link:hover,
.record-link:focus-visible {
  text-decoration: underline;
  text-underline-offset: 0.18em;
}

.samples-link {
  font-weight: 650;
}

.record-details {
  position: relative;
}

.record-details summary {
  cursor: pointer;
}

.details-content {
  position: absolute;
  z-index: 1;
  inset-inline-end: 0;
  min-width: min(19rem, calc(100vw - 5rem));
  margin-top: 0.3rem;
  padding: 0.65rem 0.8rem;
  border: 1px solid var(--c-surface-border);
  border-radius: 0.45rem;
  background: var(--c-surface);
  color: var(--c-text-secondary);
  box-shadow: 0 0.25rem 0.75rem rgb(0 0 0 / 0.16);
}

.details-content dt {
  font-weight: 650;
}

.details-content dd {
  margin-bottom: 0.3rem;
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
    gap: 0.35rem 0.75rem;
    padding: 0.65rem 0.8rem;
  }

  .catalog-table tbody th,
  .catalog-table tbody td {
    padding: 0;
  }

  .identity-cell {
    grid-column: 1 / -1;
  }

  .record-actions {
    justify-content: end;
    padding-inline-end: 0.15rem;
  }
}
</style>
