# naming — detector prompt

## AGENTS.md anchor

§ Naming — "No `*_utils.py` — use descriptive names like `text_cleaning.py`.
Function names should reflect return types. No `_s` suffix for seconds. No
abbreviations like `exe` — use `exec` or full words." § Code Style — function
names should reflect what they return.

## What to look for

Flag names that mislead readers about purpose or return type, abbreviations
that obscure intent, modules named `*_utils.py`, names that duplicate
existing type names, and vestigial qualifiers (`_v2`, `_new`, `_via_rpc`,
`_legacy_`, `_compat`) on the only surviving variant. Per `lint.md`
precedence: vestigial qualifiers on a name whose other variant is destined
for deletion file under `dead-code`; this detector owns the case where both
variants live but the qualifier no longer disambiguates.

## Anchor examples

- **Misleading return type**: `cpu_wall_ms` suggests captured CPU time, but
  actually measures task wall-clock time. Should rename to `task_wall_ms` to
  match semantics.
  ```python
  cpu_wall_ms = task_wall_time - start_time  # Actually wall time, not CPU time
  ```

- **Generic `_utils` module**:
  ```python
  # lib/marin/src/marin/processing/tokenize/tokenize_utils.py  # bad
  # lib/marin/src/marin/processing/tokenize/tokenize_helpers.py  # bad too
  # lib/marin/src/marin/processing/tokenize/byte_pair_encoding.py  # good
  ```

- **Function name mismatch with behavior**: `labeled_lm_eval` suggests a
  model called "labeled_lm" when the function does general masked-span
  evaluation unrelated to "labeled" data.
  ```python
  def labeled_lm_eval(model, data):  # Generic logic, misleading name
  ```

- **Field/type name dissonance**: `_log_client` field name when type is
  `LogClient` (imported from `finelog.client`) — the underscore prefix on a
  field that simply holds the public type is needless cognitive friction.
  ```python
  _log_client: LogClient  # field is the type; drop the underscore
  ```

- **Abbreviated parameter name**: Function `_list_staged_files` can be
  simplified to use descriptive top-level helper instead of inline logic
  with unclear intent.
  ```python
  def _list_staged_files(input_path: str):  # use existing fsspec_glob helper
  ```

- **Vestigial qualifier on the only variant**: A method or class repeats a
  word ("reconcile_workers_via_reconcile") or carries a qualifier ("_v2",
  "_new", "_legacy_", "_compat") that no longer disambiguates because the
  contrast it referred to is gone.
  ```python
  def reconcile_workers_via_reconcile(...): ...  # stutter; just reconcile_workers
  attempt_id_compat: str                          # "compat" with what?
  ```
  Why: vestigial qualifiers tell readers there are two variants when there
  is only one, and they propagate. (If the *other* variant still exists but
  is flag-gated for removal, `dead-code` owns that finding.)

## False-positive guidance

- Names matching imported type symbols are acceptable when the field
  directly holds that type (e.g., `_log_client: LogClient` may stay if the
  underscore reflects intentional private scope, not name-collision avoidance).
- Domain-specific abbreviations like `MAP` / `REDUCE` in enums are standard
  and acceptable.
- Module `utils` / `helpers` suffixes are acceptable when the module
  genuinely provides cross-cutting utilities for a large number of callers
  across the package — not local to one file or narrowly-scoped.
- A qualifier on a name where the contrasting variant *does* still exist
  (and is not flag-gated) is not vestigial; it is correctly
  disambiguating. Suppress.

## Suggested confidence floor

Require high confidence: only flag names where the mismatch between name and
semantics or behavior is unambiguous and reviewable without deep context.
Generic "this could be clearer" observations should be lower priority or
suppressed.
