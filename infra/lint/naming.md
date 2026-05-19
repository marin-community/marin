# naming — detector prompt

## What to look for
Flag function, variable, and module names that mislead readers about purpose or return type, abbreviations that obscure intent, or module names with `_utils` suffix. Also flag names that duplicate existing type names or lack descriptive specificity.

## Anchor examples

- **Misleading return type**: `cpu_wall_ms` suggests captured CPU time, but actually measures task wall-clock time. Should rename to `task_wall_ms` to match semantics.
  ```python
  cpu_wall_ms = task_wall_time - start_time  # Actually wall time, not CPU time
  ```

- **Generic `_utils` module**: Utilities modules lack descriptive names. Use concrete names like `text_cleaning.py` instead of `text_utils.py`.
  ```python
  # lib/marin/src/marin/processing/tokenize/tokenize_utils.py ❌
  # lib/marin/src/marin/processing/tokenize/tokenize_helpers.py ✓
  ```

- **Function name mismatch with behavior**: `labeled_lm_eval` suggests a model called "labeled_lm" when the function does general masked-span evaluation unrelated to "labeled" data.
  ```python
  def labeled_lm_eval(model, data):  # Generic logic, misleading name
  ```

- **Type name mismatch**: `_log_client` field name when type is `LogClient` (imported from `finelog.client`) creates unnecessary name/type cognitive dissonance where the symbol and type don't align.
  ```python
  _log_client: LogClient  # Field underscore conflicts with type name
  ```

- **Abbreviated parameter name**: Function `_list_staged_files` can be simplified to use descriptive top-level helper instead of inline logic with unclear intent.
  ```python
  def _list_staged_files(input_path: str):  # Unclear—use existing fsspec_glob helper
  ```

- **Stuttering / vestigial qualifier in name**: A method or class repeats the same word ("reconcile_workers_via_reconcile", "_reconcile_one_via_reconcile") or carries a qualifier ("_v2", "_new", "_via_rpc", "_legacy_", "_compat") that no longer disambiguates because the thing it contrasts with is gone or is the only other variant. The qualifier is vestigial branding from a transition that has completed (or never had two variants live at once).
  ```python
  # In a class that no longer has a non-reconcile path:
  def reconcile_workers_via_reconcile(...): ...  # stutter; just reconcile_workers
  def _reconcile_one_via_reconcile(...): ...    # _via_reconcile means nothing now
  # A "_compat" field that the rest of the dataclass has already standardized on:
  attempt_id_compat: str  # "compat" with what? if all callers use it, it's just attempt_id
  ```
  Why: Vestigial qualifiers tell readers there are two variants when there is only one, and they propagate (tests, call sites, docstrings all re-use the stutter). Strip the qualifier and update call sites — the cost is one rename today vs. permanent reader confusion.

## False-positive guidance

- Names matching imported type symbols are acceptable when the field directly holds that type (e.g., `_log_client: LogClient` is fine if renaming would require renaming the type too, which is a larger refactor).
- Domain-specific abbreviations like `MAP`/`REDUCE` in enums are standard and acceptable.
- Module `utils` or `helpers` suffixes are acceptable when the module genuinely provides cross-cutting utilities for a large number of callers across the package (not local to one file or narrowly-scoped).
- Comments reflecting past code state are acceptable if they explain *why* a current design was chosen (not mere historical trivia).

## Suggested confidence floor

Require high confidence: only flag names where the mismatch between name and semantics or behavior is unambiguous and reviewable without deep context. Generic "this could be clearer" observations should be lower priority.
