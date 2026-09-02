<script setup lang="ts">
import Shell from '@marina/Shell.vue'
</script>

<template>
  <Shell app="echo">
    <template #nav>
      <router-link to="/">Search</router-link>
      <router-link to="/wiki">Wiki</router-link>
      <router-link to="/conversation">Conversation</router-link>
      <router-link to="/feedback">Feedback</router-link>
      <a href="/echo/api/docs">API</a>
    </template>
    <div class="mx-auto max-w-6xl px-5 pb-16 pt-8 sm:px-8 sm:pt-10">
      <router-view />
    </div>
  </Shell>
</template>
