# imports — detector prompt

## What to look for

Flag code that defines imports inside function or class bodies, except where justified to break circular dependencies or guard optional/conditional dependencies. Per AGENTS.md: "All imports at the top of the file. No local imports except to break circular dependencies or guard optional deps."

## Anchor examples

1. **Reviewer (ravwojdyla)**: "nit: local import"
   - Code shape: Import statement inside function body in lib/zephyr/src/zephyr/external_sort.py
   - Why it matters: Local imports add cognitive overhead and make module-level dependencies implicit; they should live at the top unless there's a cycle or the module is optional.

2. **Reviewer (dlwh)**: "Moved `zstandard` to module scope and dropped the local import. This is just style cleanup; the tar.zst path still behaves the same."
   - Code shape: `import zstandard` was moved from inside a function to module-level in lib/marin/src/marin/datakit/download/formal_methods_evals.py
   - Why it matters: Standard dependencies (not optional) should always be imported at module scope for clarity and consistency.

3. **Reviewer (dlwh)**: "I moved `load_dataset` to the module-level imports and left `_load_hf_iterable` as a thin wrapper over the filtered parquet file list."
   - Code shape: `from datasets import load_dataset` moved from inside function to top-level in lib/marin/src/marin/transform/evaluation/raw_lm_eval.py
   - Why it matters: Even imports used by helper functions should be top-level unless they are truly optional or create a cycle.

4. **Reviewer (rjpower)**: "nit local imports"
   - Code shape: Multiple imports inside functions in lib/marin/src/marin/evaluation/served_lm_eval.py
   - Why it matters: Consolidating at module level improves readability and makes dependencies explicit.

5. **Reviewer (yonromai)**: "`lm_eval` is only installed with Marin's eval extra; so moving the imports to module scope makes the default marin-tests environment fail at collection with `ModuleNotFoundError`. Simplest here is to keep the lm_eval imports local?"
   - Code shape: Optional dependency imported inside function; moving to top-level breaks base test environment
   - Why it matters: This is the legitimate exception — guard optional dependencies inside functions to avoid import errors when the extra is not installed.

## False-positive guidance

- **Optional/conditional dependencies**: If a module is only available in certain install extras (e.g., `lm_eval` only with `eval` extra), local imports to guard `ImportError` are acceptable.
- **Circular dependency resolution**: If two modules import each other, a local import in one function can break the cycle. This is a legitimate use.
- **Lazy loading for performance**: Very rare, but if a module is expensive to import and only used in one rarely-called path, a local import may be justified—document it.
- **Platform-specific imports**: Code that imports only on certain OS/Python versions can use local imports with guards.
- **Test setup code**: Fixtures and test utilities may import locally if it keeps test isolation clear, but prefer top-level for consistency.

## Suggested confidence floor

Treat local imports as likely issues unless the code has an explicit try/except for ImportError or a comment explaining why (circular dep, optional extra, etc.). High confidence on bare local imports of standard library or always-installed packages.
