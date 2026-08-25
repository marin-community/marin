<script setup lang="ts">
/**
 * Full value of one cell, plus the rest of its row.
 *
 * A drawer rather than a modal: the row you clicked stays visible behind it, so
 * a wide result can be read one column at a time without losing your place.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { ColumnKind } from '@/utils/columnKind'
import { formatTimestampMs, timeZoneLabel } from '@/utils/formatting'
import { timeZoneMode } from '@/composables/useDisplayPrefs'

const props = defineProps<{
  column: string
  row: Record<string, unknown>
  kinds: Record<string, ColumnKind>
}>()

const emit = defineEmits<{ close: [] }>()

const copyState = ref<'idle' | 'copied' | 'failed'>('idle')
const activeColumn = ref(props.column)

watch(() => props.column, (next) => { activeColumn.value = next })

const raw = computed(() => props.row[activeColumn.value])
const kind = computed<ColumnKind>(() => props.kinds[activeColumn.value] ?? 'text')

/** The text put on the clipboard and shown in the body — always the full value. */
const text = computed(() => {
  const value = raw.value
  if (value === null || value === undefined) return ''
  if (kind.value === 'json' && typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), null, 2)
    } catch {
      return value
    }
  }
  return String(value)
})

const annotation = computed(() => {
  if (kind.value !== 'timestamp' || typeof raw.value !== 'number') return null
  return `${formatTimestampMs(raw.value, 'utc')} UTC · ${formatTimestampMs(raw.value, 'local')} ${timeZoneLabel('local')}`
})

const otherColumns = computed(() => Object.keys(props.row).filter((c) => c !== activeColumn.value))

/**
 * A clipboard write can be refused — an insecure origin, or a denied permission —
 * and the refusal is the only thing that tells the reader the value is not on
 * their clipboard, so it reaches the button rather than the console.
 */
async function copy() {
  try {
    await navigator.clipboard.writeText(text.value)
    copyState.value = 'copied'
  } catch {
    copyState.value = 'failed'
  }
  setTimeout(() => { copyState.value = 'idle' }, 1400)
}

const copyLabel = computed(() => ({ idle: 'Copy', copied: 'Copied', failed: 'Copy failed' })[copyState.value])

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}

onMounted(() => document.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" :aria-label="`Value of ${activeColumn}`">
    <div class="absolute inset-0 bg-black/25" @click="emit('close')" />
    <aside class="relative z-10 w-full max-w-2xl h-full bg-surface border-l border-surface-border flex flex-col shadow-2xl">
      <header class="px-5 py-3 border-b border-surface-border flex items-center gap-3">
        <div class="min-w-0">
          <div class="font-mono text-sm truncate">{{ activeColumn }}</div>
          <div class="text-xs text-text-muted">{{ kind }}</div>
        </div>
        <div class="ml-auto flex items-center gap-2">
          <button
            class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
            :class="copyState === 'failed' && 'text-status-danger border-status-danger-border'"
            @click="copy"
          >{{ copyLabel }}</button>
          <button
            class="text-xs px-2 py-1 rounded border border-surface-border hover:bg-surface-raised"
            aria-label="Close"
            @click="emit('close')"
          >Close</button>
        </div>
      </header>

      <div class="flex-1 overflow-auto">
        <p v-if="annotation" class="px-5 pt-4 text-xs text-text-secondary font-mono">{{ annotation }}</p>
        <pre
          v-if="text"
          class="px-5 py-4 font-mono text-[13px] leading-relaxed whitespace-pre-wrap break-words"
        >{{ text }}</pre>
        <p v-else class="px-5 py-4 text-sm text-text-muted">This cell is empty.</p>

        <div v-if="otherColumns.length" class="border-t border-surface-border">
          <h2 class="px-5 pt-4 pb-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">Rest of row</h2>
          <dl class="px-5 pb-5 grid grid-cols-[minmax(0,10rem)_1fr] gap-x-4 gap-y-1.5 text-[13px]">
            <template v-for="c in otherColumns" :key="c">
              <dt class="font-mono text-text-secondary truncate" :title="c">{{ c }}</dt>
              <dd class="font-mono truncate">
                <button
                  class="text-left w-full truncate hover:text-accent"
                  :title="String(row[c] ?? '')"
                  @click="activeColumn = c"
                >{{
                  kinds[c] === 'timestamp' && typeof row[c] === 'number'
                    ? formatTimestampMs(row[c] as number, timeZoneMode)
                    : (row[c] ?? '—')
                }}</button>
              </dd>
            </template>
          </dl>
        </div>
      </div>
    </aside>
  </div>
</template>
