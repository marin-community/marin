<script setup>
import { computed } from "vue";

import { markdown } from "../lib.js";

const props = defineProps({ text: String });
const html = computed(() => markdown(props.text));
</script>

<template>
  <div class="prose" v-html="html"></div>
</template>
