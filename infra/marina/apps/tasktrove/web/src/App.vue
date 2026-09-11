<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import Shell from '@marina/Shell.vue'
</script>

<template>
  <Shell app="tasktrove">
    <template #nav>
      <RouterLink to="/">Cleanup report</RouterLink>
      <RouterLink to="/#sources">Source audit</RouterLink>
      <RouterLink to="/browse">Browse tasks</RouterLink>
    </template>
    <div class="page">
      <RouterView />
    </div>
  </Shell>
</template>
