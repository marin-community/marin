<script setup lang="ts">
import { RouterView } from 'vue-router'
import AppHeader from '@/components/layout/AppHeader.vue'
import TabNav from '@/components/layout/TabNav.vue'
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <AppHeader />
    <TabNav />
    <main class="flex-1 px-6 py-4 max-w-[1400px] w-full mx-auto">
      <RouterView />
    </main>
  </div>
</template>
