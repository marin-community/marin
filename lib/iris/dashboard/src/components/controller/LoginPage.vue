<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { setAuthToken } from '@/composables/useRpc'

const router = useRouter()
const token = ref('')
const error = ref<string | null>(null)

function login() {
  const trimmed = token.value.trim()
  if (!trimmed) {
    error.value = 'Please enter a token.'
    return
  }
  setAuthToken(trimmed)
  error.value = null
  router.push('/')
}
</script>

<template>
  <div class="min-h-screen bg-surface-raised flex items-center justify-center px-4">
    <div class="w-full max-w-md">
      <div class="bg-surface border border-surface-border rounded-xl p-8 shadow-sm">
        <h1 class="text-xl font-semibold text-text mb-2">Iris Dashboard</h1>
        <p class="text-sm text-text-muted mb-6">
          This cluster requires authentication. Paste a bearer token to continue.
        </p>

        <div
          v-if="error"
          class="mb-4 px-4 py-3 text-sm text-status-danger bg-status-danger-bg rounded-lg border border-status-danger-border"
        >
          {{ error }}
        </div>

        <form @submit.prevent="login" class="space-y-4">
          <div>
            <label for="token" class="block text-sm font-medium text-text mb-1.5">Bearer Token</label>
            <textarea
              id="token"
              v-model="token"
              rows="4"
              class="w-full rounded-lg border border-surface-border bg-surface-raised px-3 py-2 text-sm font-mono text-text placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-accent focus:border-transparent resize-none"
              placeholder="Paste your token here..."
            />
          </div>
          <button
            type="submit"
            class="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent/90 transition-colors"
          >
            Login
          </button>
        </form>

        <p class="mt-6 text-xs text-text-muted leading-relaxed">
          Get a token with:
          <code class="font-mono bg-surface-raised px-1.5 py-0.5 rounded text-text text-xs">
            gcloud auth print-identity-token --audiences=AUDIENCE
          </code>
        </p>
      </div>
    </div>
  </div>
</template>
