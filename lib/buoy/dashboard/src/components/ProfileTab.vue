<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useProfile } from '../composables/useProfile'
import type { RunRef } from '../composables/useRun'

const props = defineProps<{ runRef: RunRef }>()
const { state, error, prepare } = useProfile()

onMounted(() => prepare(props.runRef))

// The same-origin /wrap frame loads the xprof reverse-proxy; it absorbs xprof's
// window.parent.history writes so they don't clobber the SPA URL. Relative path
// so it resolves under the proxy sub-path.
const wrapSrc = computed(() => {
  const r = props.runRef
  return `wrap/${encodeURIComponent(r.entity)}/${encodeURIComponent(r.project)}/${encodeURIComponent(r.run_id)}`
})
</script>

<template>
  <div class="h-[82vh]">
    <div v-if="state === 'preparing'" class="flex h-full items-center justify-center">
      <div class="text-center">
        <div class="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-4 border-surface-border border-t-accent"></div>
        <div class="text-text-secondary">preparing profile (downloading + launching xprof)…</div>
      </div>
    </div>
    <div v-else-if="state === 'error'" class="text-status-danger">profile failed: {{ error }}</div>
    <iframe v-else :src="wrapSrc" class="h-full w-full border-0" title="xprof profile"></iframe>
  </div>
</template>
