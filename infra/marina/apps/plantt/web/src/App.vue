<script setup>
import { RouterLink, RouterView } from "vue-router";

import Shell from "@marina/Shell.vue";
</script>

<template>
  <Shell app="plantt">
    <template #nav>
      <RouterLink to="/">Charts</RouterLink>
    </template>
    <RouterView />
  </Shell>
</template>
