<script setup lang="ts">
const tabs = [
  { to: '/', label: 'Pipeline graph' },
  { to: '/metrics', label: 'Metrics' },
  { to: '/counters', label: 'Counters' },
  { to: '/workers', label: 'Workers' },
]
</script>

<template>
  <nav class="border-b border-surface-border bg-surface-raised">
    <div class="mx-auto flex max-w-[1600px] gap-7 px-4 sm:px-6 lg:px-8">
      <RouterLink
        v-for="tab in tabs"
        :key="tab.to"
        :to="tab.to"
        class="border-b-2 border-transparent py-3 text-sm font-medium text-text-secondary hover:text-text"
        active-class="!border-accent !text-accent"
        :exact="tab.to === '/'"
      >
        {{ tab.label }}
      </RouterLink>
    </div>
  </nav>
</template>
