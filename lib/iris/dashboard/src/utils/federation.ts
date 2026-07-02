// Helpers for deep-linking to a federation peer's own dashboard — the one place
// the parent's mirror of a peer stops. A federated job runs on the peer under a
// deterministic rebased `remoteJobId`; the peer's dashboard renders it natively
// (with working logs and attempt history), so we link straight there.

/** Join a peer's dashboard base URL with a hash-router path (leading slash). */
function peerDashboardUrl(dashboardUrl: string, hashPath: string): string | undefined {
  const base = (dashboardUrl ?? '').trim()
  if (!base) return undefined
  // The dashboard uses createWebHashHistory, so routes live under `#/...`.
  return `${base.replace(/\/+$/, '')}/#${hashPath}`
}

/**
 * Deep-link to a federated job's page on its peer cluster's dashboard.
 * Returns undefined when either the peer dashboard URL or the remote id is
 * missing, so callers can fall back to plain text.
 */
export function peerJobUrl(dashboardUrl?: string, remoteJobId?: string): string | undefined {
  if (!dashboardUrl || !remoteJobId) return undefined
  return peerDashboardUrl(dashboardUrl, `/job/${encodeURIComponent(remoteJobId)}`)
}

/**
 * Link to a peer cluster's dashboard home. Used where the exact remote job id
 * isn't available (e.g. the task page, whose TaskStatus carries no remoteJobId),
 * so the user still has a one-click path to the peer's own dashboard.
 */
export function peerHomeUrl(dashboardUrl?: string): string | undefined {
  if (!dashboardUrl) return undefined
  return peerDashboardUrl(dashboardUrl, '/')
}
