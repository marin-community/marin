<script setup lang="ts">
/** Display an agentic sample's ATIF trajectory or retained SOTOPIA episode. */
import { ref, watch } from 'vue'
import { apiGet } from '@/composables/useApi'
import type { ArtifactResponse, SampleRow, Trajectory } from '@/types/api'
import { isSotopiaSummary, type SotopiaSummary as SotopiaSummaryData } from '@/utils/sotopia'
import EmptyState from '@/components/shared/EmptyState.vue'
import SotopiaSummary from '@/components/samples/SotopiaSummary.vue'
import TrajectoryViewer from '@/components/samples/TrajectoryViewer.vue'

const props = defineProps<{ sample: SampleRow; runId: string }>()

const loading = ref(false)
const error = ref<string | null>(null)
const unavailableReason = ref<string | null>(null)
const trajectory = ref<Trajectory | null>(null)
const sotopiaSummary = ref<SotopiaSummaryData | null>(null)
let generation = 0

async function load(runId: string, uri: string | null) {
  const currentGeneration = ++generation
  trajectory.value = null
  sotopiaSummary.value = null
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
      unavailableReason.value = artifact.reason ?? 'Sample artifact is unavailable.'
      return
    }
    const parsed: unknown = JSON.parse(artifact.text)
    if (isSotopiaSummary(parsed)) sotopiaSummary.value = parsed
    else if (typeof parsed === 'object' && parsed !== null && 'steps' in parsed && Array.isArray(parsed.steps)) {
      trajectory.value = parsed as Trajectory
    } else {
      throw new Error('Unsupported sample artifact format.')
    }
  } catch (e) {
    if (currentGeneration !== generation) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (currentGeneration === generation) loading.value = false
  }
}

watch(
  () => [props.runId, props.sample.summary_uri ?? props.sample.trajectory_uri] as const,
  ([runId, uri]) => load(runId, uri),
  { immediate: true },
)
</script>

<template>
  <section class="space-y-2">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-text-secondary">
      {{ sotopiaSummary ? 'SOTOPIA episode' : 'Trajectory' }}
    </h3>

    <EmptyState v-if="!sample.summary_uri && !sample.trajectory_uri" message="No episode artifact recorded for this sample." icon="○" />
    <p v-else-if="loading" class="text-sm text-text-muted py-6 text-center">Loading episode…</p>
    <div
      v-else-if="error"
      class="rounded border border-status-danger-border bg-status-danger-bg text-status-danger text-sm px-3 py-2"
    >{{ error }}</div>
    <p
      v-else-if="unavailableReason"
      class="rounded border border-status-warning-border bg-status-warning-bg text-status-warning text-sm px-3 py-2"
    >Episode unavailable — {{ unavailableReason }}</p>
    <SotopiaSummary v-else-if="sotopiaSummary" :summary="sotopiaSummary" />
    <TrajectoryViewer v-else-if="trajectory" :trajectory="trajectory" />
  </section>
</template>
