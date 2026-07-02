/**
 * Composable that fetches /auth/config once (module-level singleton) and
 * exposes the multi-backend roster to all tabs and components.
 *
 * The singleton pattern ensures the /auth/config request is made exactly once
 * regardless of how many components call useBackends().
 */
import { ref, computed } from 'vue'
import type { LocationQueryValue, RouteLocationNormalizedLoaded } from 'vue-router'
import { controllerRpcCall } from '@/composables/useRpc'
import type { BackendInfo, ListBackendsResponse, ListPeersResponse, PeerSummary } from '@/types/rpc'

/**
 * Resolve a scope query param to a validated id. Returns undefined when the
 * param is absent, or when a non-empty ``knownIds`` roster does not contain it.
 * An empty roster (not yet loaded) passes the id through unchecked.
 */
function resolveScopeId(
  raw: LocationQueryValue | LocationQueryValue[],
  knownIds: string[],
): string | undefined {
  const idStr = Array.isArray(raw) ? (raw[0] ?? '') : (raw ?? '')
  if (!idStr) return undefined
  if (knownIds.length > 0 && !knownIds.includes(idStr)) return undefined
  return idStr
}

// Module-level state — shared across all callers.
const backends = ref<BackendInfo[]>([])
const capabilities = ref<string[]>([])
// Federation peers this cluster can hand jobs off to. Populated from ListPeers
// (the /auth/config payload carries backends but not peers). Empty on a
// single-cluster deployment, so every peer-derived affordance stays inert.
const peers = ref<PeerSummary[]>([])
let _configFetched = false
let _peersFetched = false

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
   * Return the `?backend=` query param value, or undefined when absent or not in
   * the (populated) backend roster — an invalid id never reaches the server as a
   * backend_id filter.
   */
  function currentBackend(route: RouteLocationNormalizedLoaded): string | undefined {
    return resolveScopeId(route.query.backend, backends.value.map(b => b.id))
  }

  /** One-shot call to the ListBackends RPC. */
  async function listBackends(): Promise<ListBackendsResponse> {
    return controllerRpcCall<ListBackendsResponse>('ListBackends', {})
  }

  /** One-shot call to the ListPeers RPC; also refreshes the shared roster. */
  async function listPeers(): Promise<ListPeersResponse> {
    const resp = await controllerRpcCall<ListPeersResponse>('ListPeers', {})
    peers.value = resp.peers ?? []
    _peersFetched = true
    return resp
  }

  /**
   * Load the peer roster once (for gates and `?cluster=` validation). Safe to
   * call from many components — only the first successful fetch performs the RPC;
   * a failure leaves the roster empty and lets a later call retry.
   */
  async function ensurePeers(): Promise<void> {
    if (_peersFetched) return
    try {
      await listPeers()
    } catch {
      // Peers unavailable (older controller / not authed) — leave roster empty.
    }
  }

  /**
   * Return the `?cluster=` query param value, or undefined when absent or not in
   * the (populated) peer roster — an invalid id never reaches the server as a
   * child_cluster filter.
   */
  function currentCluster(route: RouteLocationNormalizedLoaded): string | undefined {
    return resolveScopeId(route.query.cluster, peers.value.map(p => p.peerId))
  }

  /** The peer with this id, or undefined. Used to resolve a peer's dashboard URL. */
  function peerById(peerId: string): PeerSummary | undefined {
    return peers.value.find(p => p.peerId === peerId)
  }

  return {
    backends,
    capabilities,
    peers,
    multiBackend,
    fetchConfig,
    currentBackend,
    currentCluster,
    listBackends,
    listPeers,
    ensurePeers,
    peerById,
  }
}
