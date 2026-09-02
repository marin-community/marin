<script setup lang="ts">
// A run of prose, with the rows it names as links.
//
// A model that has just written a paragraph holds a URN, and a URN in a
// paragraph is a string nobody clicks. `mentions` finds them and `path` turns
// each one into the address of the screen that shows it.
//
// An `href` and not a router link: a URN usually names a row in another app,
// which is another bundle behind another path prefix, so following one is a
// page load either way.

import { computed } from 'vue'
import { type Mention, mentions, path } from './urn'

const props = defineProps<{ text: string }>()

const pieces = computed(() => mentions(props.text))

/** Where a piece points, empty for prose. */
function address(piece: Mention): string {
  return (piece.urn && path(piece.urn)) || ''
}
</script>

<template>
  <span class="linked">
    <template v-for="(piece, at) in pieces" :key="at">
      <a v-if="address(piece)" :href="address(piece)">{{ piece.text }}</a>
      <span v-else>{{ piece.text }}</span>
    </template>
  </span>
</template>

<style scoped>
/* `parts.css` styles an anchor and this states its own: the class layer is
 * opted into per app, and a component every app may drop in cannot require a
 * stylesheet some of them do not import. */
.linked a {
  color: var(--mark);
  text-decoration: underline;
  text-underline-offset: 0.15em;
}
</style>
