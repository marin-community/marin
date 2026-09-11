/**
 * Per-variant accelerator capacity derived from BackendSummary.availability
 * (free/total chips keyed by lowercased device-variant), plus aggregation
 * across a set of backends (a peer's forwarded roster, or the whole fleet).
 */
import type { BackendSummary } from '@/types/rpc'

export interface VariantCapacity {
  /** Lowercased device-variant token, e.g. "h100". */
  token: string
  /** Free chips right now. */
  free: number
  /** Total chips over the same capacity; null when the reporting cluster
   *  predates the total_amounts field. */
  total: number | null
}

/** Per-variant free/total for one backend, sorted by token. Empty when the
 *  backend does not supply the availability metric (proto field unset). */
export function backendCapacities(b: BackendSummary): VariantCapacity[] {
  const availability = b.availability
  if (!availability) return []
  const tokens = new Set([
    ...Object.keys(availability.amounts ?? {}),
    ...Object.keys(availability.totalAmounts ?? {}),
  ])
  return [...tokens].sort().map((token) => ({
    token,
    free: Number(availability.amounts?.[token] ?? '0'),
    total:
      availability.totalAmounts?.[token] != null ? Number(availability.totalAmounts[token]) : null,
  }))
}

/** Sum capacities by token. A variant's total goes null if any contributor
 *  omits its total — a partial sum would understate the denominator. */
export function aggregateCapacities(entries: VariantCapacity[]): VariantCapacity[] {
  const byToken = new Map<string, VariantCapacity>()
  for (const entry of entries) {
    const prior = byToken.get(entry.token)
    if (!prior) {
      byToken.set(entry.token, { ...entry })
      continue
    }
    prior.free += entry.free
    prior.total = prior.total != null && entry.total != null ? prior.total + entry.total : null
  }
  return [...byToken.values()].sort((a, b) => a.token.localeCompare(b.token))
}
