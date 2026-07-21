<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { post } from '@/api'

const router = useRouter()
const text = ref('')
const error = ref('')
const submitting = ref(false)

async function submit() {
  if (!text.value.trim() || submitting.value) return
  submitting.value = true
  try {
    const response = await post<{ case_id: string }>('/api/questions', { text: text.value })
    await router.push(`/cases/${response.case_id}`)
  } catch (caught) {
    error.value = caught instanceof Error ? caught.message : String(caught)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <section class="mx-auto max-w-3xl">
    <p class="mb-1 font-mono text-[11px] uppercase tracking-[0.18em] text-[#657169]">Manual investigation</p>
    <h2 class="text-3xl font-semibold tracking-tight">Ask the ops agent</h2>
    <p class="mt-2 text-sm leading-6 text-[#68726c]">
      One-off questions use the same durable queue, read-only credentials, and visible Loom transcript as Grafana-triggered cases.
    </p>
    <form class="mt-7 rounded-xl border border-[#d2d8d0] bg-white p-5 shadow-[0_2px_8px_rgba(25,38,31,.05)]" @submit.prevent="submit">
      <label for="question" class="mb-2 block text-sm font-semibold">What should the agent investigate?</label>
      <textarea id="question" v-model="text" rows="8" autofocus class="w-full resize-y rounded-lg border border-[#cbd2ca] bg-[#fafbf9] p-4 text-sm leading-6 outline-none focus:border-[#679778] focus:ring-2 focus:ring-[#d9ecdf]" placeholder="For example: Is the image filesystem pressure on g5bea54 caused by unused images, logs, or active workloads? Validate without changing the node." />
      <div v-if="error" class="mt-3 rounded border border-[#efb6b2] bg-[#fff1ef] p-2 text-sm text-[#8e2822]">{{ error }}</div>
      <div class="mt-4 flex items-center justify-between">
        <p class="text-xs text-[#7d8680]">The agent may read Kubernetes and Iris state. It cannot mutate production.</p>
        <button type="submit" :disabled="submitting || !text.trim()" class="rounded-lg bg-[#1c5b3a] px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40">{{ submitting ? 'Queuing…' : 'Start investigation' }}</button>
      </div>
    </form>
  </section>
</template>
