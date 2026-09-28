# Follow-up review: Delphi TPP40 one-pair bridge

Review the same files and frozen artifacts named in `.agents/handoffs/delphi_tpp40_one_pair_bridge_cc_review_20260830.md` after the first review's fixes.

The first review reported two blockers:

1. Recursive `sort_keys=True` reordered `component_bpb`, while the analyzer required insertion order.
2. `all([])` made `--allow-incomplete` suppress a fully materialized numerical failure.

Changes made:

- The Uncheatable validator now compares the exact component-name set, then reconstructs canonical order through `_uncheatable_metrics`.
- `_allow_incomplete_failure` requires a nonempty set of loading errors, or independently permits only the all-numerical-pass/missing-idempotence state.
- A real write/read round-trip regression test covers sorted nested JSON keys.
- Tests cover complete numerical failure not being masked by `--allow-incomplete`.
- The stale `--max-concurrent=8` default now derives from the one-row/two-checkpoint cell count.
- Path-manifest validation now checks the training-output count.
- Idempotence evidence no longer self-asserts skipped/executed counts. The analyzer verifies succeeded rerun records with zero child jobs, exact command hashes, unchanged before/after/current inventories, and measured current unit counts.

Local evidence after the fixes:

- 71 focused tests pass.
- `uv run pyrefly check`: zero errors.
- Targeted repository pre-commit: pass.
- All four exact launch commands still pass region-local launch-safety validation.

Return `GO` only if both blockers are fixed and the strengthened idempotence checks remain internally consistent. Otherwise return `NO-GO` with exact file and line references. Separate production blockers from polish.
