# docs — detector prompt

## What to look for
Flag code with outdated, incomplete, or misleading documentation. This includes docstrings describing old behavior, comments that restate code without explaining why, and functions documented as read-only but mutating state.

## Anchor examples

1. **Docstring describes old behavior**
   - ravwojdyla: "docstring not updated. `size` here feels a bit off, `InputFileSpec` would imply 'specification', isn't `size` metadata not spec?"
   - Code shape: Parameter documented with one meaning but semantic has shifted.
   - Why: Readers trust docstrings; stale docs cause incorrect usage.

2. **Non-obvious return value undocumented**
   - ravwojdyla: "In the docstring we could describe what the return bool holds since it's non-obvious here without reading the code."
   - Code shape: Function returns boolean with semantic meaning not obvious from name.
   - Why: Callers must read implementation to understand semantics; docstring prevents that.

3. **Outdated inline comment**
   - ravwojdyla: "Old comment was outdated, we are doing character level shingling NOT word level."
   - Code shape: Inline comment describing algorithm or behavior from previous version.
   - Why: Comments help future readers reason about code; outdated comments mislead about intent.

4. **Documentation contradicts implementation**
   - yonromai: "`run_corpus_mode()` documented as read-only, but opening the directory through live `DuckDBLogStore` is not passive."
   - Code shape: Docstring claim (e.g., "read-only") but code mutates state.
   - Why: Callers relying on documented contract suffer data corruption or side effects.

5. **Missing explanation for bespoke behavior**
   - dlwh: "since these are bespoke we should explain what they are"
   - Code shape: Non-standard configuration or special-case logic without clarification.
   - Why: Maintainers cannot understand why unconventional patterns exist.

6. **Stale reference in external documentation**
   - RohithKuditipudi: "updated a stale reference in OPS.md"
   - Code shape: Code updated but linked docs (OPS.md, mkdocs) reference old API/behavior.
   - Why: Operational docs are trusted; stale cross-references break runbooks.

7. **Rotting historical references in source comments/docstrings**
   - Reviewer language: "this comment will rot the moment X ships", "drop the phase reference", "don't reference kata short codes in source".
   - Code shape: Docstrings or inline comments that name a rollout phase ("Phase B+", "Phase C"), a kata/issue short code ("kata h9r9"), a migration number ("see migration 0047 which added this column"), a PR number ("introduced in #4321"), or a branch/sprint name. These are scaffolding vocabulary that mean nothing once the work lands.
   - Why: Time-bound references decay; a reader six months later cannot recover the context, and the comment becomes actively misleading. Either link to a durable explanation (an ADR, a module-level invariant) or remove the reference. Acceptable forms are durable identifiers (a permanent design doc path, a stable issue URL), not rolling project vocabulary.

## False-positive guidance

- **Docstring updates in same PR**: If PR itself adds/refreshes docstring to match new code, not a nit.
- **Comments added for clarity**: Comments explaining subtle behavior in same commit are acceptable.
- **Placeholder docs**: Code is new and docs follow in later PR (unless public API).
- **Self-explanatory code**: Simple functions (getters, wrappers) with clear names don't need docstrings per AGENTS.md.
- **Internal-only code**: Temporary helpers (`_foo`) don't require docstrings unless complex or cross-module.

## Suggested confidence floor

Flag only if comment explicitly names documentation issue (docstring, comment, outdated, stale, explain, non-obvious) or points out docs-behavior contradiction. Avoid flagging behavior nits mentioning docs as context.
