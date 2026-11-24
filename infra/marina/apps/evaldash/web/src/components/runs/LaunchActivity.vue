<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, type RouteLocationRaw } from 'vue-router'
import ModelName from '@/components/shared/ModelName.vue'
import StatusChip from '@/components/shared/StatusChip.vue'
import { MIXED_LAUNCH_STATUS, type LaunchRow } from '@/types/api'
import { formatTimestamp } from '@/utils/formatting'

const props = defineProps<{
  rows: LaunchRow[]
  launchLocation: (group: string) => RouteLocationRaw
}>()

const dayFormat = new Intl.DateTimeFormat(undefined, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' })
const timeFormat = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })
const numberFormat = new Intl.NumberFormat()

interface LaunchEntry {
  row: LaunchRow
  description: string
  time: string
  suffix?: string
}

const MINIMUM_ID_SUFFIX = 4

function distinguishMatchingEntries(entries: LaunchEntry[]) {
  const byAppearance = new Map<string, LaunchEntry[]>()
  for (const entry of entries) {
    const { row, description, time } = entry
    const title = description || row.model_name || row.version || row.group_id
    const key = JSON.stringify([title, row.model_name, row.version, row.user_name, row.accelerator, time, row.status, runCount(row)])
    const matches = byAppearance.get(key) ?? []
    if (!byAppearance.has(key)) byAppearance.set(key, matches)
    matches.push(entry)
  }
  for (const matches of byAppearance.values()) {
    if (matches.length < 2) continue
    let length = MINIMUM_ID_SUFFIX
    while (new Set(matches.map(({ row }) => row.group_id.slice(-length))).size < matches.length) length++
    for (const entry of matches) entry.suffix = entry.row.group_id.slice(-length)
  }
}

const dateGroups = computed(() => {
  const groups = new Map<string, LaunchEntry[]>()
  for (const row of props.rows) {
    const date = new Date(row.created_at)
    const valid = !Number.isNaN(date.getTime())
    const day = valid ? dayFormat.format(date) : 'Date unavailable'
    const items = groups.get(day) ?? []
    if (!groups.has(day)) groups.set(day, items)
    items.push({ row, description: row.description?.trim() ?? '', time: valid ? timeFormat.format(date) : '' })
  }
  return Array.from(groups, ([day, items]) => {
    distinguishMatchingEntries(items)
    return { day, items }
  })
})

function runCount(row: LaunchRow) {
  const count = numberFormat.format(row.n_evals)
  if (row.status === MIXED_LAUNCH_STATUS) return `${numberFormat.format(row.n_succeeded)} of ${count} succeeded`
  return `${count} ${row.n_evals === 1 ? 'run' : 'runs'}`
}
</script>

<template>
  <div class="launch-activity" role="region" aria-label="Evaluation launches">
    <section
      v-for="({ day, items }, dayIndex) in dateGroups"
      :key="day"
      class="launch-group"
      :aria-labelledby="`launch-day-${dayIndex}`"
    >
      <h2 :id="`launch-day-${dayIndex}`" class="launch-day">{{ day }}</h2>
      <ol class="launch-list" role="list" :aria-labelledby="`launch-day-${dayIndex}`">
        <li v-for="({ row, description, time, suffix }, rowIndex) in items" :key="row.group_id" class="launch-entry">
          <RouterLink
            :to="launchLocation(row.group_id)"
            class="launch-link"
            :aria-labelledby="`launch-title-${dayIndex}-${rowIndex}`"
            :title="row.group_id"
          >
            <time v-if="time" class="launch-time" :datetime="row.created_at" :title="formatTimestamp(row.created_at)">{{ time }}</time>
            <div class="launch-identity">
              <h3 :id="`launch-title-${dayIndex}-${rowIndex}`" class="launch-title">
                <span v-if="description">{{ description }}</span>
                <ModelName v-else-if="row.model_name" :model="row.model_name" />
                <span v-else>{{ row.version || row.group_id }}</span>
                <span v-if="suffix" class="launch-id">
                  <code aria-hidden="true">#{{ suffix }}</code>
                  <span class="sr-only">Launch ID {{ row.group_id }}</span>
                </span>
              </h3>
              <div class="launch-scope">
                <ModelName v-if="description && row.model_name" :model="row.model_name" />
                <span v-if="row.version && (description || row.model_name)" :title="`Version ${row.version}`">{{ row.version }}</span>
                <span v-if="row.user_name" :title="`Owner ${row.user_name}`">{{ row.user_name }}</span>
                <span v-if="row.accelerator" :title="`Accelerator ${row.accelerator}`">{{ row.accelerator }}</span>
              </div>
            </div>
            <div class="launch-outcome">
              <StatusChip :status="row.status" />
              <span>{{ runCount(row) }}</span>
            </div>
            <span class="launch-arrow" aria-hidden="true">→</span>
          </RouterLink>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.launch-activity {
  padding-inline: clamp(0.65rem, 2vw, 1rem);
}

.launch-group {
  display: grid;
  grid-template-columns: 9.75rem minmax(0, 1fr);
  gap: 0.4rem;
}

.launch-group + .launch-group {
  border-top: 1px solid var(--c-surface-border);
}

.launch-day {
  padding-block-start: 0.9rem;
  color: var(--c-text-secondary);
  font-size: 0.82rem;
  font-weight: 650;
}

.launch-entry + .launch-entry {
  border-top: 1px solid var(--c-surface-border-subtle);
}

.launch-link {
  display: grid;
  grid-template-areas: 'time identity outcome arrow';
  grid-template-columns: 4.65rem minmax(0, 1fr) 12.5rem 1rem;
  align-items: center;
  gap: 0.6rem;
  min-height: 2.75rem;
  padding: 0.42rem 0.55rem;
}

.launch-link:hover,
.launch-link:focus-visible {
  background: color-mix(in srgb, var(--c-accent) 6%, var(--c-surface));
}

.launch-time,
.launch-scope,
.launch-outcome {
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
}

.launch-time {
  grid-area: time;
  white-space: nowrap;
}

.launch-identity {
  display: flex;
  grid-area: identity;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.05rem 0.6rem;
  min-width: 0;
  overflow-wrap: anywhere;
}

.launch-title {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.25rem;
  min-width: 0;
  color: var(--c-accent);
  font-size: 0.9375rem;
  font-weight: 650;
  line-height: 1.35;
}

.launch-id {
  color: var(--c-text-secondary);
  font-size: 0.8125rem;
  font-weight: 500;
}

.launch-link:hover .launch-title,
.launch-link:focus-visible .launch-title {
  text-decoration: underline;
  text-underline-offset: 0.16em;
}

.launch-scope {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.1rem 0.45rem;
}

.launch-scope > span + span::before {
  content: '·';
  display: inline-block;
  margin-inline-end: 0.45rem;
  color: var(--c-text-muted);
}

.launch-outcome {
  display: flex;
  grid-area: outcome;
  align-items: center;
  justify-content: space-between;
  gap: 0.36rem;
  white-space: nowrap;
}

.launch-arrow {
  grid-area: arrow;
  color: var(--c-accent);
  text-align: end;
}

@media (max-width: 790px) {
  .launch-group {
    display: block;
  }

  .launch-day {
    padding-block: 0.52rem 0.28rem;
  }

  .launch-link {
    grid-template-areas: 'identity identity arrow' 'time outcome arrow';
    grid-template-columns: minmax(0, 1fr) auto 1rem;
    gap: 0.28rem 0.5rem;
    padding: 0.5rem 0.3rem;
  }
}
</style>
