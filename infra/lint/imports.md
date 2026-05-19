# imports — detector prompt

## AGENTS.md anchor

§ Code Style — "All imports at the top of the file. No local imports except
to break circular dependencies or guard optional deps. No `TYPE_CHECKING`
guards — fix cycles structurally via protocols."

## What to look for

Flag imports defined inside function or class bodies, and `TYPE_CHECKING`
guard blocks. The two legitimate exceptions are (a) breaking a real import
cycle, and (b) guarding an optional dependency that is only installed with a
specific extra. Both exceptions should be obvious from surrounding code (a
`try/except ImportError`, a comment naming the cycle, or a docstring noting
the extra).

## Anchor examples

1. **Bare local import of a standard dependency** (ravwojdyla): "nit: local
   import" on a line inside a function body in
   `lib/zephyr/src/zephyr/external_sort.py`. Standard always-installed
   modules should live at module scope.

2. **Local import of a third-party that's already a hard dep** (dlwh):
   "Moved `zstandard` to module scope and dropped the local import. This is
   just style cleanup; the tar.zst path still behaves the same."

3. **Helper-only local imports** (dlwh): `from datasets import load_dataset`
   moved from inside a function to top-level in
   `lib/marin/src/marin/transform/evaluation/raw_lm_eval.py`. Even imports
   used by helper functions should be top-level unless they are truly
   optional or create a cycle.

4. **Multiple stacked local imports** (rjpower): "nit local imports" on a
   block of imports inside functions in
   `lib/marin/src/marin/evaluation/served_lm_eval.py`. Consolidate at module
   level.

5. **Legitimate optional-dep guard** (yonromai): "`lm_eval` is only installed
   with Marin's eval extra; so moving the imports to module scope makes the
   default marin-tests environment fail at collection with
   `ModuleNotFoundError`. Simplest here is to keep the lm_eval imports
   local?" This is the canonical exception — surface it as confirmation
   rather than a finding.

6. **`TYPE_CHECKING` block**: AGENTS.md forbids these outright; restructure
   with a `Protocol` in the layer that owns the type.

## False-positive guidance

- **Optional/conditional dependencies**: If a module is only available in
  certain install extras (e.g., `lm_eval` only with `eval` extra), local
  imports guarded by `try/except ImportError` or wrapped in a clearly-named
  function are correct. Do not flag.
- **Circular dependency resolution**: If two modules import each other, a
  local import in one function can break the cycle. Accept it but suggest a
  `Protocol`-based fix if the cycle looks structurally avoidable.
- **Lazy loading for startup performance**: Very rare, but if a module is
  expensive to import and only used in one rarely-called path, a local
  import may be justified — require a one-line comment naming the reason.
- **Platform-specific imports**: Code that imports only on certain
  OS/Python versions can use local imports with guards.

## Suggested confidence floor

High confidence on a bare local import of a standard-library or
always-installed package with no surrounding `try/except ImportError` and
no nearby comment explaining the placement. Lower confidence — or
suppression — when the import is inside a function whose name or docstring
makes the optional-dep / cycle-break intent explicit. Note: a legitimate
optional-dep guard is *correct code*, not a `nit`; either suppress or
emit at the lowest confidence and explain why in the message.
