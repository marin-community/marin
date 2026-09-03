<script setup lang="ts">
/**
 * The log panel's time control: a relative window, an absolute start instant,
 * and the clock both are read on, behind one trigger.
 *
 * It replaces a native `datetime-local` input, which could not accept a
 * timestamp pasted out of a log line and demanded all seven of its segments
 * before it would commit anything. The two ways an operator actually arrives at
 * a start time are pasting one they are already looking at and stepping away
 * from one they already have, so this offers a tolerant text field and coarse
 * nudges either side of it.
 */
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { SINCE_PRESETS } from '@/composables/useTimeWindow'
import { formatLogTimestamp, parseLogTimestamp, type TimeZoneName } from '@/utils/formatting'
import { CONTROL, CONTROL_OFF, CONTROL_ON, FIELD } from '@/styles/controls'

const props = defineProps<{
  presetMs: number
  sinceInstant: number | null
  timeZone: TimeZoneName
}>()

const emit = defineEmits<{
  selectPreset: [ms: number]
  setSince: [ms: number]
  setTimeZone: [zone: TimeZoneName]
}>()

/** Steps offered either side of the current start instant, coarsest last. */
const NUDGES = [
  { label: '1m', ms: 60_000 },
  { label: '15m', ms: 15 * 60_000 },
  { label: '1h', ms: 3_600_000 },
]
const EARLIER = [...NUDGES].reverse()

const ZONES: { value: TimeZoneName; label: string }[] = [
  { value: 'local', label: 'Local' },
  { value: 'utc', label: 'UTC' },
]

const open = ref(false)
const root = ref<HTMLElement | null>(null)
const field = ref<HTMLInputElement | null>(null)
// The field holds raw text while it is being edited; only a value that parses
// is promoted to an instant, so the panel never queries a half-typed bound.
const draft = ref('')
const invalid = ref(false)

const label = computed(() => {
  const zone = props.timeZone === 'utc' ? ' · UTC' : ''
  if (props.sinceInstant === null) {
    const preset = SINCE_PRESETS.find((p) => p.ms === props.presetMs)
    return `${preset ? preset.label : 'All time'}${zone}`
  }
  const stamp = formatLogTimestamp(props.sinceInstant, props.timeZone)
  // Drop today's date: the timestamp column beside it is a clock too.
  const today = formatLogTimestamp(Date.now(), props.timeZone).slice(0, 10)
  return `Since ${stamp.startsWith(today) ? stamp.slice(11) : stamp}${zone}`
})

function syncDraft() {
  draft.value = props.sinceInstant === null ? '' : formatLogTimestamp(props.sinceInstant, props.timeZone)
  invalid.value = false
}

watch(() => [props.sinceInstant, props.timeZone] as const, syncDraft)

function applyDraft() {
  const text = draft.value.trim()
  if (!text) {
    // Emptying the field hands the window back to the relative preset.
    if (props.sinceInstant !== null) emit('selectPreset', props.presetMs)
    invalid.value = false
    return
  }
  const ms = parseLogTimestamp(text, props.timeZone)
  invalid.value = ms === null
  if (ms !== null) emit('setSince', ms)
}

/** Step the start instant, seeding from now when there is not one yet. */
function nudge(deltaMs: number) {
  emit('setSince', (props.sinceInstant ?? Date.now()) + deltaMs)
}

function choosePreset(ms: number) {
  emit('selectPreset', ms)
  open.value = false
}

function toggle() {
  open.value = !open.value
  if (!open.value) return
  syncDraft()
  nextTick(() => field.value?.focus())
}

function onPointerDown(event: MouseEvent) {
  if (!open.value) return
  if (!root.value?.contains(event.target as Node)) open.value = false
}

document.addEventListener('mousedown', onPointerDown)
onUnmounted(() => document.removeEventListener('mousedown', onPointerDown))
</script>

<template>
  <div ref="root" class="relative" @keydown.esc.stop="open = false">
    <button
      data-log-since
      type="button"
      :class="[CONTROL, open || sinceInstant !== null ? CONTROL_ON : CONTROL_OFF]"
      :title="sinceInstant === null ? 'Lower time bound for the log query' : 'Log query starts at this instant'"
      @click="toggle"
    >
      <span class="tabular-nums">{{ label }}</span>
      <span class="text-text-muted text-xs">▾</span>
    </button>

    <div
      v-if="open"
      class="absolute left-0 z-20 mt-1 w-72 space-y-3 rounded-lg border border-surface-border
             bg-surface p-3 shadow-lg"
    >
      <div class="grid grid-cols-3 gap-1">
        <button
          v-for="preset in SINCE_PRESETS"
          :key="preset.ms"
          type="button"
          class="h-6 rounded border px-1 text-xs"
          :class="sinceInstant === null && presetMs === preset.ms ? CONTROL_ON : CONTROL_OFF"
          @click="choosePreset(preset.ms)"
        >{{ preset.label }}</button>
      </div>

      <div class="space-y-1">
        <div class="flex items-baseline justify-between">
          <span class="text-xs text-text-secondary">Start at</span>
          <button type="button" class="text-xs text-accent hover:underline" @click="emit('setSince', Date.now())">
            Now
          </button>
        </div>
        <input
          ref="field"
          v-model="draft"
          data-log-since-field
          type="text"
          spellcheck="false"
          placeholder="2026-08-20 11:03:22.480"
          :class="[FIELD, 'h-7 w-full text-sm', invalid ? 'border-status-danger' : '']"
          @keyup.enter="applyDraft"
          @blur="applyDraft"
        />
        <p class="text-xs" :class="invalid ? 'text-status-danger' : 'text-text-muted'">
          {{ invalid ? 'Unrecognized time' : 'Date, time, or both. Epoch seconds and ms too.' }}
        </p>
      </div>

      <div class="flex items-center gap-1">
        <button
          v-for="step in EARLIER"
          :key="`earlier-${step.ms}`"
          type="button"
          class="h-6 flex-1 rounded border text-xs tabular-nums"
          :class="CONTROL_OFF"
          :title="`Move the start ${step.label} earlier`"
          @click="nudge(-step.ms)"
        >−{{ step.label }}</button>
        <span class="mx-1 h-4 w-px shrink-0 bg-surface-border" />
        <button
          v-for="step in NUDGES"
          :key="`later-${step.ms}`"
          type="button"
          class="h-6 flex-1 rounded border text-xs tabular-nums"
          :class="CONTROL_OFF"
          :title="`Move the start ${step.label} later`"
          @click="nudge(step.ms)"
        >+{{ step.label }}</button>
      </div>

      <div class="flex items-center justify-between border-t border-surface-border-subtle pt-2">
        <span class="text-xs text-text-secondary">Timestamps</span>
        <div class="inline-flex gap-1">
          <button
            v-for="zone in ZONES"
            :key="zone.value"
            type="button"
            class="h-6 rounded border px-2 text-xs"
            :class="timeZone === zone.value ? CONTROL_ON : CONTROL_OFF"
            @click="emit('setTimeZone', zone.value)"
          >{{ zone.label }}</button>
        </div>
      </div>
    </div>
  </div>
</template>
