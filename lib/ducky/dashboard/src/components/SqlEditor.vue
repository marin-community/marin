<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { Compartment, EditorState, Prec } from '@codemirror/state'
import { EditorView, keymap, placeholder } from '@codemirror/view'
import { autocompletion } from '@codemirror/autocomplete'
import { basicSetup } from 'codemirror'
import { PostgreSQL, sql } from '@codemirror/lang-sql'
import { oneDark } from '@codemirror/theme-one-dark'

const props = defineProps<{ modelValue: string; dark?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [string]; run: [] }>()

const host = ref<HTMLElement | null>(null)
let view: EditorView | null = null
const themeComp = new Compartment()

// Dark mode uses oneDark (readable light-on-dark palette + syntax colors); light mode
// uses the app surface tokens. basicSetup's default highlight is tuned for light, so
// it only needs a background/gutter tweak there.
const lightTheme = EditorView.theme({
  '&': { backgroundColor: 'var(--c-surface-raised)' },
  '.cm-gutters': { backgroundColor: 'var(--c-surface-sunken)', color: 'var(--c-text-muted)', border: 'none' },
})

function editorTheme(dark: boolean | undefined) {
  return dark ? oneDark : lightTheme
}

const PLACEHOLDER = `-- write DuckDB SQL, then ⌘/Ctrl-Enter to run
SELECT *
FROM read_parquet('gs://marin-us-east5/<path>/*.parquet')
LIMIT 100`

onMounted(() => {
  view = new EditorView({
    parent: host.value!,
    state: EditorState.create({
      doc: props.modelValue,
      extensions: [
        basicSetup,
        sql({ dialect: PostgreSQL }), // DuckDB SQL is close to Postgres
        // completion stays available on Ctrl-Space, but doesn't pop up on every
        // keystroke (the default was an intrusive symbol dropdown while typing)
        autocompletion({ activateOnTyping: false }),
        placeholder(PLACEHOLDER),
        EditorView.lineWrapping,
        // high precedence so Mod-Enter runs the query instead of inserting a newline
        Prec.highest(
          keymap.of([
            {
              key: 'Mod-Enter',
              run: () => {
                emit('run')
                return true
              },
            },
          ]),
        ),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) emit('update:modelValue', u.state.doc.toString())
        }),
        themeComp.of(editorTheme(props.dark)),
        EditorView.theme({
          '&': { fontSize: '13px' },
          '&.cm-focused': { outline: 'none' },
          '.cm-content': { fontFamily: 'var(--font-mono, monospace)' },
        }),
      ],
    }),
  })
})

// Swap the editor theme when the app toggles dark mode.
watch(
  () => props.dark,
  (dark) => view?.dispatch({ effects: themeComp.reconfigure(editorTheme(dark)) }),
)

// Reflect external model changes (e.g. clicking a catalog view/example to pre-fill) into
// the editor. Guard against the echo of our own updateListener emit, which would reset the
// cursor on every keystroke.
watch(
  () => props.modelValue,
  (next) => {
    if (!view || next === view.state.doc.toString()) return
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: next } })
  },
)

onBeforeUnmount(() => view?.destroy())
</script>

<template>
  <div ref="host" class="overflow-hidden rounded-lg border border-surface-border"></div>
</template>
