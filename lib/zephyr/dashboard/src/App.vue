<script setup lang="ts">
import { RouterView } from 'vue-router'
import AppHeader from '@/components/AppHeader.vue'
import TabNav from '@/components/TabNav.vue'
</script>

<template>
  <div class="min-h-screen bg-surface">
    <AppHeader />
    <TabNav />
    <main class="mx-auto w-full max-w-[1600px] px-4 py-5 sm:px-6 lg:px-8">
      <RouterView />
    </main>
  </div>
</template>
