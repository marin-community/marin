<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { EditorState, Prec } from '@codemirror/state'
import { EditorView, keymap, placeholder } from '@codemirror/view'
import { basicSetup } from 'codemirror'
import { PostgreSQL, sql } from '@codemirror/lang-sql'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits<{ 'update:modelValue': [string]; run: [] }>()

const host = ref<HTMLElement | null>(null)
let view: EditorView | null = null

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
        EditorView.theme({
          '&': { fontSize: '13px', backgroundColor: 'var(--c-surface-raised)' },
          '&.cm-focused': { outline: 'none' },
          '.cm-content': { fontFamily: 'var(--font-mono, monospace)' },
          '.cm-gutters': {
            backgroundColor: 'var(--c-surface-sunken)',
            color: 'var(--c-text-muted)',
            border: 'none',
          },
        }),
      ],
    }),
  })
})

onBeforeUnmount(() => view?.destroy())
</script>

<template>
  <div ref="host" class="overflow-hidden rounded-lg border border-surface-border"></div>
</template>
