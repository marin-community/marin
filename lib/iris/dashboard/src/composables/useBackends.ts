/**
 * Composable that fetches /auth/config once (module-level singleton) and
 * exposes the multi-backend roster to all tabs and components.
 *
 * The singleton pattern ensures the /auth/config request is made exactly once
 * regardless of how many components call useBackends().
 */
import { ref, computed } from 'vue'
import type { RouteLocationNormalizedLoaded } from 'vue-router'
import { controllerRpcCall } from '@/composables/useRpc'
import type { BackendInfo, ListBackendsResponse } from '@/types/rpc'

// Module-level state — shared across all callers.
const backends = ref<BackendInfo[]>([])
const capabilities = ref<string[]>([])
let _configFetched = false

export interface AuthConfig {
  authEnabled: boolean
  hasSession: boolean
  authOptional: boolean
}

export function useBackends() {
  const multiBackend = computed(() => backends.value.length > 1)

  /**
   * Fetch /auth/config and populate the module-level singleton.
   * Safe to call multiple times — only the first call performs the fetch.
   * Returns auth-related fields so App.vue can handle login redirection
   * without a second fetch.
   */
  async function fetchConfig(): Promise<AuthConfig> {
    const authDefaults: AuthConfig = { authEnabled: false, hasSession: false, authOptional: false }
    if (_configFetched) return authDefaults
    _configFetched = true
    try {
      const resp = await fetch('/auth/config')
      if (!resp.ok) return authDefaults
      const config = await resp.json() as {
        auth_enabled?: boolean
        has_session?: boolean
        optional?: boolean
        capabilities?: string[]
        backends?: Array<{ id: string; name?: string; capabilities?: string[] }>
        backend?: { capabilities?: string[] }
      }
      // Use the union capabilities served by the updated /auth/config, falling
      // back to the legacy single-backend field so a pre-feature-PR controller
      // still gates tabs correctly.
      capabilities.value = config.capabilities ?? config.backend?.capabilities ?? []
      if (Array.isArray(config.backends) && config.backends.length > 0) {
        backends.value = config.backends.map(b => ({
          id: b.id,
          name: b.name ?? b.id,
          capabilities: b.capabilities ?? [],
        }))
      }
      return {
        authEnabled: config.auth_enabled ?? false,
        hasSession: config.has_session ?? false,
        authOptional: config.optional ?? false,
      }
    } catch {
      // Endpoint unavailable — leave capabilities/backends empty.
      return authDefaults
    }
  }

  /**
   * Return the `?backend=` query param value, or undefined when absent/invalid.
   * An unknown id (not in the roster) is silently cleared so callers never
   * pass an invalid backend_id filter to the server.
   */
  function currentBackend(route: RouteLocationNormalizedLoaded): string | undefined {
    const id = route.query.backend
    const idStr = Array.isArray(id) ? (id[0] ?? '') : (id ?? '')
    if (!idStr) return undefined
    // If the roster is populated, validate the id.
    if (backends.value.length > 0 && !backends.value.find(b => b.id === idStr)) {
      return undefined
    }
    return idStr
  }

  /** One-shot call to the ListBackends RPC. */
  async function listBackends(): Promise<ListBackendsResponse> {
    return controllerRpcCall<ListBackendsResponse>('ListBackends', {})
  }

  return {
    backends,
    capabilities,
    multiBackend,
    fetchConfig,
    currentBackend,
    listBackends,
  }
}
