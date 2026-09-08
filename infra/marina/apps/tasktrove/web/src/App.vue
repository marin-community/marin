<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import Shell from '@marina/Shell.vue'
</script>

<template>
  <Shell app="tasktrove">
    <template #nav>
      <RouterLink to="/">Sources</RouterLink>
      <RouterLink to="/sampled">Audited sample</RouterLink>
      <RouterLink to="/find">Find a task</RouterLink>
    </template>
    <div class="page">
      <p class="where">open-thoughts/TaskTrove, read live from Hugging Face</p>
      <RouterView />
    </div>
  </Shell>
</template>
