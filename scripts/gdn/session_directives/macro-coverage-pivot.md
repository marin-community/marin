# Session Directive: Coverage Pivot To Hybrid-Specific Shell Delta, A3, And P3

Goal:
- keep `3/4` GDN fixed,
- fully demote same-boundary GDN hillclimbing from the mainline,
- spend iteration budget on the hybrid-specific shell delta, the AD-boundary prototype, and the first serious P3 prototype.

Coverage rule for this session:
- `S3` is already satisfied on the current xprof-enabled harness.
- Do not spend another mainline iteration on `S3` unless the iteration itself changes xprof / attribution plumbing.
- Before spending another mainline iteration on same-boundary GDN shell/tape/kernel work,
  complete at least one validated attempt for each of:
  - `A3` AD-boundary prototype,
  - `P3` fixed-4-layer block prototype.

Diagnostic allowance:
- CE work is allowed only when fresh attribution re-implicates CE.
- same-boundary GDN work is diagnostic only unless it directly reduces `hybrid_generic_shell_delta_budget_ms`.
