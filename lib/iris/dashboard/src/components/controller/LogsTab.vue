<script setup lang="ts">
import LogViewer from '@/components/shared/LogViewer.vue'
</script>

<template>
  <div class="space-y-4">
    <p class="text-sm text-text-secondary">
      Cluster-wide log explorer. Narrow by source key — a user (<code class="font-mono">/alice/</code>),
      a job (<code class="font-mono">/alice/job/</code>), a task attempt
      (<code class="font-mono">/alice/job/0:1</code>), or a worker
      (<code class="font-mono">/system/worker/…</code>) — then filter by text, level, and time.
      Task and worker ids in each line are clickable.
    </p>
    <LogViewer standalone max-height="72vh" />
  </div>
</template>
