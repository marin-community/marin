<script setup lang="ts">
/**
 * SQL textarea with identifier completion.
 *
 * A textarea rather than a code editor: the value of completion here is not
 * knowing SQL, it is knowing *this store's* namespaces and columns, which no
 * general editor can supply. Keeping the input a plain textarea also keeps
 * selection, undo, and mobile keyboards behaving the way the browser already
 * makes them behave.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { completionsFor, tokenAt, type Completion, type NamespaceColumns } from '@/utils/sqlComplete'

const props = defineProps<{
  modelValue: string
  schema: NamespaceColumns[]
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
  submit: []
}>()

const textarea = ref<HTMLTextAreaElement | null>(null)
const suggestions = ref<Completion[]>([])
const active = ref(0)
const caretBox = ref({ left: 0, top: 0 })
/** Suppressed until the next edit, so Escape dismisses without re-opening. */
const dismissed = ref(false)

const open = computed(() => suggestions.value.length > 0 && !dismissed.value)

/**
 * Pixel position of the caret inside the textarea.
 *
 * A textarea exposes no caret geometry, so the text up to the caret is laid out
 * again in a hidden div that copies the textarea's own type metrics; the marker
 * span's offset is where the caret sits.
 */
function measureCaret(el: HTMLTextAreaElement): { left: number; top: number } {
  const style = window.getComputedStyle(el)
  const mirror = document.createElement('div')
  for (const prop of [
    'fontFamily', 'fontSize', 'fontWeight', 'letterSpacing', 'lineHeight',
    'paddingTop', 'paddingLeft', 'paddingRight', 'borderTopWidth', 'borderLeftWidth',
    'boxSizing', 'whiteSpace', 'wordWrap', 'tabSize',
  ] as const) {
    mirror.style[prop] = style[prop]
  }
  mirror.style.position = 'absolute'
  mirror.style.visibility = 'hidden'
  mirror.style.whiteSpace = 'pre-wrap'
  mirror.style.wordWrap = 'break-word'
  mirror.style.width = `${el.clientWidth}px`
  mirror.textContent = el.value.slice(0, el.selectionStart)
  const marker = document.createElement('span')
  marker.textContent = '​'
  mirror.appendChild(marker)
  document.body.appendChild(mirror)
  const left = marker.offsetLeft
  const top = marker.offsetTop + parseFloat(style.lineHeight || '16')
  document.body.removeChild(mirror)
  return { left: left - el.scrollLeft, top: top - el.scrollTop }
}

function refresh() {
  const el = textarea.value
  if (!el) return
  const caret = el.selectionStart
  const { text } = tokenAt(props.modelValue, caret)
  // An empty token would offer the whole vocabulary on every click into the
  // editor, which is noise rather than help.
  suggestions.value = text.length >= 1 ? completionsFor(props.modelValue, caret, props.schema) : []
  active.value = 0
  if (suggestions.value.length) caretBox.value = measureCaret(el)
}

function onInput(e: Event) {
  emit('update:modelValue', (e.target as HTMLTextAreaElement).value)
  dismissed.value = false
  void nextTick(refresh)
}

/** Replace the token under the caret with `choice`, and put the caret after it. */
function accept(choice: Completion) {
  const el = textarea.value
  if (!el) return
  const caret = el.selectionStart
  const { start } = tokenAt(props.modelValue, caret)
  const next = props.modelValue.slice(0, start) + choice.insert + props.modelValue.slice(caret)
  emit('update:modelValue', next)
  suggestions.value = []
  void nextTick(() => {
    el.focus()
    const at = start + choice.insert.length
    el.setSelectionRange(at, at)
  })
}

function onKeydown(e: KeyboardEvent) {
  if (open.value) {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      const step = e.key === 'ArrowDown' ? 1 : suggestions.value.length - 1
      active.value = (active.value + step) % suggestions.value.length
      return
    }
    if (e.key === 'Tab' || (e.key === 'Enter' && !e.ctrlKey && !e.metaKey)) {
      e.preventDefault()
      accept(suggestions.value[active.value])
      return
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      dismissed.value = true
      return
    }
  }
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault()
    suggestions.value = []
    emit('submit')
  }
}

// A schema that arrives after the first keystroke should still complete.
watch(() => props.schema, () => {
  if (document.activeElement === textarea.value) refresh()
})

defineExpose({ focus: () => textarea.value?.focus() })
</script>

<template>
  <div class="relative">
    <textarea
      ref="textarea"
      :value="modelValue"
      class="w-full font-mono text-sm bg-surface-sunken border border-surface-border rounded p-3 min-h-[120px] focus:outline-none focus:border-accent"
      spellcheck="false"
      autocomplete="off"
      autocapitalize="off"
      @input="onInput"
      @keydown="onKeydown"
      @blur="suggestions = []"
      @click="dismissed = true"
    />
    <ul
      v-if="open"
      class="absolute z-20 min-w-[16rem] max-h-64 overflow-auto rounded border border-surface-border bg-surface shadow-lg py-1 text-xs"
      :style="{ left: `${caretBox.left}px`, top: `${caretBox.top}px` }"
      role="listbox"
    >
      <li
        v-for="(s, i) in suggestions"
        :key="`${s.kind}:${s.value}`"
        class="px-2.5 py-1 flex items-baseline gap-2 cursor-pointer"
        :class="i === active ? 'bg-accent text-white' : 'hover:bg-surface-raised'"
        role="option"
        :aria-selected="i === active"
        @mousedown.prevent="accept(s)"
        @mouseenter="active = i"
      >
        <span class="font-mono">{{ s.value }}</span>
        <span class="ml-auto shrink-0" :class="i === active ? 'text-white/70' : 'text-text-muted'">
          {{ s.detail }}
        </span>
      </li>
    </ul>
  </div>
</template>
