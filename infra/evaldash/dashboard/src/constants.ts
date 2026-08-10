// Most models the Compare surface scores at once. The panel picker and the compare page
// share this so they never disagree on how many models are sendable.
export const MAX_COMPARE = 4

// Run properties a panel or comparison request can be filtered on. Mirrors metrics.RUN_FACETS; the
// server rejects nothing it does not recognize, it simply ignores it, so drift is inert.
export const FACETS = ['accelerator', 'platform', 'backend', 'mechanism', 'user'] as const
