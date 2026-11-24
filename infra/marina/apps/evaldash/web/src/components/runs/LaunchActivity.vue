<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import StatusChip from '@/components/shared/StatusChip.vue'
import type { LaunchRow } from '@/types/api'
import { formatTimestamp } from '@/utils/formatting'

const props = defineProps<{
  rows: LaunchRow[]
  launchLocation: (group: string) => RouteLocationRaw
}>()

const dayFormat = new Intl.DateTimeFormat(undefined, { dateStyle: 'full' })
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })
const numberFormat = new Intl.NumberFormat()

const dateGroups = computed(() => {
  const groups = new Map<string, { row: LaunchRow; description: string; time: string }[]>()
  for (const row of props.rows) {
    const date = new Date(row.created_at)
    const valid = !Number.isNaN(date.getTime())
    const day = valid ? dayFormat.format(date) : 'Date unavailable'
    const items = groups.get(day) ?? []
    if (!groups.has(day)) groups.set(day, items)
    items.push({ row, description: row.description?.trim() ?? '', time: valid ? timeFormat.format(date) : '' })
  }
  return Array.from(groups, ([day, items]) => ({ day, items }))
})

function runCount(row: LaunchRow) {
  const count = numberFormat.format(row.n_evals)
  if (row.status === 'mixed') return `${numberFormat.format(row.n_succeeded)} of ${count} succeeded`
  return `${count} ${row.n_evals === 1 ? 'run' : 'runs'}`
}
</script>

<template>
  <div class="launch-activity" role="region" aria-label="Evaluation launches">
    <section v-for="({ day, items }, index) in dateGroups" :key="day" :aria-labelledby="`launch-day-${index}`">
      <h2 :id="`launch-day-${index}`" class="launch-day">{{ day }}</h2>
      <ol class="launch-list" role="list" :aria-labelledby="`launch-day-${index}`">
        <li v-for="{ row, description, time } in items" :key="row.group_id" class="launch-entry">
          <time v-if="time" class="launch-time" :datetime="row.created_at" :title="formatTimestamp(row.created_at)">{{ time }}</time>
          <div class="launch-identity">
            <h3 class="launch-title">
              <span v-if="description" :title="description">{{ description }}</span>
              <ModelName v-else-if="row.model_name" :model="row.model_name" />
              <span v-else>{{ row.version || row.group_id }}</span>
            </h3>
            <div v-if="description && row.model_name" class="launch-model">
              <ModelName :model="row.model_name" />
            </div>
            <div v-if="row.version || row.user_name || row.accelerator" class="launch-scope">
              <span v-if="row.version && (description || row.model_name)" :title="`Version ${row.version}`">{{ row.version }}</span>
              <span v-if="row.user_name" :title="`Owner ${row.user_name}`">{{ row.user_name }}</span>
              <span v-if="row.accelerator" :title="`Accelerator ${row.accelerator}`">{{ row.accelerator }}</span>
            </div>
          </div>
          <div class="launch-outcome">
            <StatusChip :status="row.status" />
            <span class="launch-count">{{ runCount(row) }}</span>
          </div>
          <RouterLink
            :to="launchLocation(row.group_id)"
            class="launch-link"
            :aria-label="`View runs for ${row.model_name || row.version || row.group_id}`"
            :title="row.group_id"
          >View runs <span aria-hidden="true">→</span></RouterLink>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.launch-activity {
  display: grid;
  gap: 1.5rem;
  padding: 1.2rem clamp(0.65rem, 2vw, 1.25rem) 1.4rem;
}

.launch-day {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.6rem;
  color: var(--c-text-secondary);
  font-size: 0.82rem;
  font-weight: 650;
}

.launch-day::after {
  content: '';
  flex: 1;
  border-top: 1px solid var(--c-surface-border);
}

.launch-list {
  display: grid;
  gap: 0.5rem;
}

.launch-entry {
  display: grid;
  grid-template-areas: 'when identity outcome action';
  grid-template-columns: 5rem minmax(0, 1fr) minmax(7.5rem, auto) auto;
  align-items: start;
  gap: 0.85rem;
  padding: 0.85rem 1rem;
  border: 1px solid var(--c-surface-border);
  border-radius: 0.55rem;
  background: color-mix(in srgb, var(--c-surface-raised) 35%, var(--c-surface));
}

.launch-entry:hover,
.launch-entry:focus-within {
  border-color: var(--c-accent-border);
}

.launch-time {
  grid-area: when;
  margin-top: 0.15rem;
  color: var(--c-text-secondary);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.launch-identity {
  grid-area: identity;
  min-width: 0;
  overflow-wrap: anywhere;
}

.launch-title {
  color: var(--c-text);
  font-size: 1rem;
  font-weight: 660;
  line-height: 1.35;
}

.launch-model {
  margin-top: 0.2rem;
  color: var(--c-text);
  font-size: 0.85rem;
}

.launch-scope {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.15rem 0.65rem;
  margin-top: 0.25rem;
  color: var(--c-text-secondary);
  font-size: 0.75rem;
}

.launch-scope span + span::before {
  content: '·';
  display: inline-block;
  margin-inline-end: 0.65rem;
  color: var(--c-text-muted);
}

.launch-outcome {
  display: grid;
  justify-items: start;
  gap: 0.15rem;
  grid-area: outcome;
}

.launch-count {
  color: var(--c-text-secondary);
  font-size: 0.75rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.launch-link {
  display: inline-flex;
  grid-area: action;
  align-items: center;
  gap: 0.35rem;
  justify-self: end;
  min-height: 2rem;
  padding-inline: 0.55rem;
  border: 1px solid var(--c-surface-border);
  border-radius: 0.4rem;
  background: var(--c-surface);
  color: var(--c-accent);
  font-size: 0.78rem;
  font-weight: 650;
  white-space: nowrap;
}

.launch-link:hover,
.launch-link:focus-visible {
  border-color: var(--c-accent-border);
  background: var(--c-accent-subtle);
}

@media (max-width: 710px) {
  .launch-entry {
    grid-template-areas:
      'when outcome'
      'identity identity'
      'action action';
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 0.45rem 0.7rem;
    padding: 0.7rem 0.8rem;
  }

  .launch-outcome {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.22rem 0.45rem;
    justify-content: end;
  }
}

@media (pointer: coarse) {
  .launch-link {
    min-height: 2.75rem;
  }
}
</style>
