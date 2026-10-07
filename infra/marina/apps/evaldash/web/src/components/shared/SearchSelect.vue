<script setup lang="ts">
import { computed, nextTick, ref, useId } from 'vue'

const props = defineProps<{
  label: string
  options: string[]
  modelValue: string
}>()
const emit = defineEmits<{ 'update:modelValue': [string] }>()
const id = useId()
const input = ref<HTMLInputElement | null>(null)
const list = ref<HTMLElement | null>(null)
const open = ref(false)
const query = ref('')
const active = ref(-1)

const matches = computed(() => {
  const terms = query.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
  return props.options.filter((option) => terms.every((term) => option.toLowerCase().includes(term)))
})

function startSearch() {
  query.value = ''
  active.value = -1
  open.value = true
}

function choose(value: string) {
  emit('update:modelValue', value)
  open.value = false
}

function search(event: Event) {
  query.value = (event.target as HTMLInputElement).value
  active.value = -1
  open.value = true
}

async function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    open.value = false
    event.preventDefault()
    return
  }
  if (event.key === 'Enter' && open.value) {
    event.preventDefault()
    const option = matches.value[active.value < 0 ? 0 : active.value]
    if (option) choose(option)
    return
  }
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  event.preventDefault()
  if (!open.value) startSearch()
  if (!matches.value.length) return
  const direction = event.key === 'ArrowDown' ? 1 : -1
  active.value = active.value < 0
    ? (direction === 1 ? 0 : matches.value.length - 1)
    : (active.value + direction + matches.value.length) % matches.value.length
  await nextTick()
  list.value?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
}

function clear() {
  choose('')
  input.value?.focus()
}

function toggle() {
  if (open.value) open.value = false
  else {
    input.value?.focus()
    startSearch()
  }
}
</script>

<template>
  <div class="relative min-w-0">
    <label :for="id" class="block text-xs text-text-secondary mb-1">{{ label }}</label>
    <div class="relative">
      <input
        :id="id"
        ref="input"
        role="combobox"
        autocomplete="off"
        aria-autocomplete="list"
        :aria-expanded="open"
        :aria-controls="`${id}-options`"
        :aria-activedescendant="open && active >= 0 ? `${id}-option-${active}` : undefined"
        :value="open ? query : modelValue"
        :placeholder="`Search ${label.toLowerCase()}…`"
        :title="modelValue || undefined"
        class="w-full rounded border border-surface-border bg-surface pl-2 pr-16 py-1 text-sm"
        @focus="startSearch"
        @click="!open && startSearch()"
        @input="search"
        @keydown="onKeydown"
        @blur="open = false"
      />
      <button
        v-if="modelValue"
        type="button"
        :aria-label="`Clear ${label.toLowerCase()} filter`"
        class="absolute right-7 top-0 bottom-0 px-1.5 text-text-muted hover:text-text"
        @click="clear"
      >×</button>
      <button
        type="button"
        :aria-label="`Browse ${label.toLowerCase()} options`"
        :aria-expanded="open"
        :aria-controls="`${id}-options`"
        class="absolute right-1 top-0 bottom-0 px-1.5 text-text-muted hover:text-text"
        @mousedown.prevent
        @click="toggle"
      >▾</button>
    </div>
    <ul
      v-if="open"
      :id="`${id}-options`"
      ref="list"
      role="listbox"
      :aria-label="label"
      class="absolute z-30 mt-1 w-full min-w-[18rem] max-h-72 overflow-y-auto rounded border border-surface-border bg-surface shadow-lg"
    >
      <li
        v-for="(option, index) in matches"
        :id="`${id}-option-${index}`"
        :key="option"
        role="option"
        :aria-selected="active === index"
        class="cursor-pointer px-3 py-2 text-sm break-words hover:bg-surface-raised flex items-start gap-2"
        :class="active === index ? 'bg-accent-subtle text-accent' : 'text-text'"
        @mousedown.prevent
        @click="choose(option)"
      >
        <span class="w-4 shrink-0" aria-hidden="true">{{ option === modelValue ? '✓' : '' }}</span>
        <span class="min-w-0">{{ option }}</span>
      </li>
      <li v-if="!matches.length" role="presentation" class="px-3 py-3 text-sm text-text-muted">
        No matches. Try a shorter name.
      </li>
    </ul>
  </div>
</template>
