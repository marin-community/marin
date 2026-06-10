<script setup lang="ts">
import { RouterLink } from 'vue-router'

const tabs = [
  { to: '/', label: 'Namespaces' },
  { to: '/query', label: 'Query' },
  { to: '/logs', label: 'Logs' },
]
</script>

<template>
  <nav class="border-b border-surface-border bg-surface px-6">
    <div class="flex gap-1">
      <RouterLink
        v-for="t in tabs"
        :key="t.to"
        :to="t.to"
        v-slot="{ isActive }"
        custom
      >
        <a
          :href="t.to"
          class="px-3 py-2 text-sm border-b-2 -mb-px transition-colors"
          :class="isActive
            ? 'border-accent text-text font-medium'
            : 'border-transparent text-text-muted hover:text-text'"
          @click.prevent="$router.push(t.to)"
        >{{ t.label }}</a>
      </RouterLink>
    </div>
  </nav>
</template>
