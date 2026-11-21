<script setup>
// Pretty-printed JSON with copy and a download of the stored file.
import { computed } from "vue";

import { copyText, pretty } from "../lib.js";

const props = defineProps({ value: null, title: String, download: String });
const text = computed(() => pretty(props.value));
</script>

<template>
  <div class="codeblock json">
    <div class="cb-head">
      <span>{{ title }}</span>
      <span class="row" style="gap: 4px">
        <button class="ghost copy" title="Copy the record to the clipboard" @click="copyText(text, `Copied ${title}`)">⧉ Copy</button>
        <a v-if="download" class="copy" :href="download" download>⤓ Download</a>
      </span>
    </div>
    <pre><code>{{ text }}</code></pre>
  </div>
</template>
