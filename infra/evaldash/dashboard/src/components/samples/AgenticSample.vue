<script setup lang="ts">
/**
 * Agentic (Harbor) sample view. The reward/grading is shown by the page's GradingPanel above; this
 * component lazy-loads the step trajectory referenced by `trajectory_uri` through the server's
 * artifact endpoint and renders it, degrading to a clear state when the URI is absent or the object
 * is unreachable. `exchange_uri` points at a prediction's raw request/response artifact; it is
 * surfaced as a note when present, and rendering its contents is out of scope here.
 */
import { ref, watch } from 'vue'
import { apiGet } from '@/composables/useApi'
import type { ArtifactResponse, SampleRow, Trajectory } from '@/types/api'
import EmptyState from '@/components/shared/EmptyState.vue'
import TrajectoryViewer from '@/components/samples/TrajectoryViewer.vue'

const props = defineProps<{ sample: SampleRow; runId: string }>()

const loading = ref(false)
const error = ref<string | null>(null)
const unavailableReason = ref<string | null>(null)
const trajectory = ref<Trajectory | null>(null)
let generation = 0

async function load(runId: string, uri: string | null) {
  const currentGeneration = ++generation
  trajectory.value = null
  error.value = null
  unavailableReason.value = null
  if (!uri) {
    loading.value = false
    return
  }
  loading.value = true
  try {
    const params = new URLSearchParams({ uri })
    const artifact = await apiGet<ArtifactResponse>(`api/runs/${runId}/samples/artifact?${params.toString()}`)
    if (currentGeneration !== generation) return
    if (!artifact.available || artifact.text === null) {
      unavailableReason.value = artifact.reason ?? 'Trajectory is unavailable.'
      return
    }
    trajectory.value = JSON.parse(artifact.text) as Trajectory
  } catch (e) {
    if (currentGeneration !== generation) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (currentGeneration === generation) loading.value = false
  }
}

watch(
  () => [props.runId, props.sample.trajectory_uri] as const,
  ([runId, uri]) => load(runId, uri),
  { immediate: true },
)
</script>

<template>
  <section class="space-y-2">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">Trajectory</h3>

    <EmptyState v-if="!sample.trajectory_uri" message="No trajectory recorded for this sample." icon="○" />
    <p v-else-if="loading" class="text-sm text-text-muted py-6 text-center">Loading trajectory…</p>
    <div
      v-else-if="error"
      class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2"
    >{{ error }}</div>
    <p
      v-else-if="unavailableReason"
      class="rounded border border-status-warning-border bg-status-warning-bg text-status-warning text-sm px-3 py-2"
    >Trajectory unavailable — {{ unavailableReason }}</p>
    <TrajectoryViewer v-else-if="trajectory" :trajectory="trajectory" />

    <p v-if="sample.exchange_uri" class="text-xs text-text-muted">
      Request/response exchange recorded for this sample.
    </p>
  </section>
</template>
