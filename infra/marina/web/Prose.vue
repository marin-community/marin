<script setup lang="ts">
// One agent message, drawn as what it is.
//
// Nothing here is `v-html`: every string reaches the page through `{{ }}`, so a
// message that contains markup shows the markup. See `markdown.ts` beside this file for why
// that matters for text a model wrote.

import { computed } from 'vue'
import { blocks, followable, type Span } from './markdown'

const props = defineProps<{ text: string }>()

const parsed = computed(() => blocks(props.text))

/** The tag for a heading, capped: an `h1` inside a transcript is not a title. */
function heading(level: number): string {
  return `h${Math.min(level + 2, 6)}`
}

/** The href to follow, or nothing, in which case the link is drawn as text. */
function href(span: Span): string | undefined {
  return span.kind === 'link' && span.href && followable(span.href) ? span.href : undefined
}
</script>

<template>
  <div class="prose">
    <template v-for="(block, index) in parsed" :key="index">
      <component :is="heading(block.level)" v-if="block.kind === 'heading'">
        <span
          v-for="(span, n) in block.spans"
          :key="n"
          :class="[span.kind, { strong: span.strong, emphasis: span.emphasis }]"
          >{{ span.text }}</span
        >
      </component>

      <pre v-else-if="block.kind === 'code'" :data-language="block.language || null"><code>{{
        block.text
      }}</code></pre>

      <component :is="block.ordered ? 'ol' : 'ul'" v-else-if="block.kind === 'list'">
        <li v-for="(item, n) in block.items" :key="n">
          <template v-for="(span, m) in item" :key="m">
            <a
              v-if="href(span)"
              :href="href(span)"
              target="_blank"
              rel="noreferrer"
              :class="{ strong: span.strong, emphasis: span.emphasis }"
              >{{ span.text }}</a
            >
            <span
              v-else
              :class="[span.kind, { strong: span.strong, emphasis: span.emphasis }]"
              >{{ span.text }}</span
            >
          </template>
        </li>
      </component>

      <blockquote v-else-if="block.kind === 'quote'">
        <span
          v-for="(span, n) in block.spans"
          :key="n"
          :class="[span.kind, { strong: span.strong, emphasis: span.emphasis }]"
          >{{ span.text }}</span
        >
      </blockquote>

      <hr v-else-if="block.kind === 'rule'" />

      <p v-else>
        <template v-for="(span, n) in block.spans" :key="n">
          <a
            v-if="href(span)"
            :href="href(span)"
            target="_blank"
            rel="noreferrer"
            :class="{ strong: span.strong, emphasis: span.emphasis }"
            >{{ span.text }}</a
          >
          <span
            v-else
            :class="[span.kind, { strong: span.strong, emphasis: span.emphasis }]"
            >{{ span.text }}</span
          >
        </template>
      </p>
    </template>
  </div>
</template>

<style scoped>
.prose { display: flex; flex-direction: column; gap: 0.5rem; min-width: 0; }

p { margin: 0; overflow-wrap: anywhere; }

h3, h4, h5, h6 {
  margin: 0.35rem 0 0;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.01em;
}

.code {
  font: 0.85em var(--mono);
  background: color-mix(in srgb, var(--muted) 12%, transparent);
  border-radius: 3px;
  padding: 0.05em 0.28em;
}
.strong { font-weight: 640; }
.emphasis { font-style: italic; }

a { color: var(--mark); }

/* A fenced block scrolls rather than wraps: a shell line broken across several
 * rows is one an operator cannot copy and read at the same time. */
pre {
  margin: 0;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--edge);
  border-radius: var(--radius);
  background: var(--panel);
  font: 0.78rem var(--mono);
  overflow-x: auto;
  max-height: 24rem;
}

ul, ol { margin: 0; padding-left: 1.2rem; display: flex; flex-direction: column; gap: 0.2rem; }
li { overflow-wrap: anywhere; }

blockquote {
  margin: 0;
  padding-left: 0.7rem;
  border-left: 2px solid var(--edge);
  color: var(--muted);
}

hr { border: none; border-top: 1px solid var(--edge); margin: 0.2rem 0; width: 100%; }
</style>
