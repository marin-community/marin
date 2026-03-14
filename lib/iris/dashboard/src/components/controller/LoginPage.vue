<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const token = ref('')
const error = ref<string | null>(null)
const loading = ref(false)

async function login() {
  const trimmed = token.value.trim()
  if (!trimmed) {
    error.value = 'Please enter a token.'
    return
  }

  error.value = null
  loading.value = true
  try {
    // Exchange the token for a JWT via Login RPC.
    // Handles raw identity tokens (static config tokens, GCP access tokens).
    // If Login is unimplemented, the token is already a JWT — use it directly.
    let sessionToken = trimmed
    try {
      const loginResp = await fetch('/iris.cluster.ControllerService/Login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identity_token: trimmed }),
      })
      if (loginResp.ok) {
        const loginData = await loginResp.json()
        if (loginData.token) {
          sessionToken = loginData.token
        }
      } else {
        // Surface auth failures (e.g. invalid token). Only fall through for
        // "unimplemented" (Login not configured) — token may already be a JWT.
        const errData = await loginResp.json().catch(() => ({}))
        const code = errData.code || ''
        if (code !== 'unimplemented') {
          throw new Error(errData.message || `Login failed (${loginResp.status})`)
        }
      }
    } catch (loginErr) {
      // Network errors (Login RPC unreachable) — try token as-is
      if (loginErr instanceof TypeError) {
        // fetch network error — ignore and try token directly
      } else {
        throw loginErr
      }
    }

    const resp = await fetch('/auth/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: sessionToken }),
    })
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}))
      throw new Error(body.error || `Failed to set session (${resp.status})`)
    }
    router.push('/')
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
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
              :disabled="loading"
            />
          </div>
          <button
            type="submit"
            :disabled="loading"
            class="w-full rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent/90 transition-colors disabled:opacity-50"
          >
            {{ loading ? 'Authenticating...' : 'Login' }}
          </button>
        </form>

        <p class="mt-6 text-xs text-text-muted leading-relaxed">
          Get a token with:
          <code class="font-mono bg-surface-raised px-1.5 py-0.5 rounded text-text text-xs">
            uv run iris --controller-url=CONTROLLER_URL login
          </code>
        </p>
      </div>
    </div>
  </div>
</template>
