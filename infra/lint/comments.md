# comments — detector prompt

## AGENTS.md anchor

§ Comments — "Write comments for module/class-level behavior or subtle logic.
Do not restate the code." § LLM-Generated Code Pitfalls — verbose/redundant
docstrings.

## What to look for

Flag comments and docstrings that paraphrase the code on the next line, narrate
what a well-named function does, or describe the current task/PR/caller rather
than a durable invariant. The bar AGENTS.md sets is "WHY non-obvious" — if
removing the comment wouldn't confuse a future reader, the comment shouldn't
exist. This detector is about *vacuous* comments; `docs` covers *stale* ones.

## Anchor examples

- **Comment restates the line below**:
  ```python
  # Increment the counter
  counter += 1
  ```
  Why: the code already says this; the comment is pure noise.

- **Docstring narrates obvious behavior**:
  ```python
  def get_user_id(user: User) -> str:
      """Return the user's id."""
      return user.id
  ```
  Why: the name, signature, and one-line body already convey this. AGENTS.md
  explicitly says skip docstrings on trivial functions with clear names.

- **Comment references the current task/PR/caller**:
  ```python
  # Added for the canary ferry flow (see PR #5712)
  retry_attempts = 3
  ```
  Why: belongs in the PR description and git blame. In source it rots — a
  reader six months later cannot recover the context and the reference becomes
  misleading.

- **Multi-paragraph docstring on a one-liner**:
  ```python
  def normalize(s: str) -> str:
      """
      Normalize a string.

      This function takes a string and returns its normalized form. The
      normalization process consists of stripping whitespace ...
      """
      return s.strip().lower()
  ```
  Why: AGENTS.md § Comments says one short line max. Multi-paragraph docstrings
  on trivial helpers are an LLM-generated pitfall.

- **`__all__` in `__init__.py` listing every public symbol**:
  ```python
  __all__ = ["Foo", "Bar", "Baz", ...]
  ```
  Why: AGENTS.md § LLM-Generated Code Pitfalls flags this as a redundancy when
  the module already exports the same names via `from .x import Foo`.

- **Comment is a TODO without an owner or condition**:
  ```python
  # TODO: clean this up
  ```
  Why: actionable TODOs name the trigger (e.g. "after the migration lands")
  or the owner. Bare TODOs accumulate; they're noise that signals work without
  enabling it.

## False-positive guidance

- **Comments on subtle invariants** are exactly what AGENTS.md wants: a hidden
  constraint, a thread-safety guarantee, a workaround for a specific upstream
  bug. Do not flag.
- **Module- or class-level docstrings** that orient the reader are encouraged
  by AGENTS.md. The bar is lower than for individual functions.
- **Public-API docstrings**: Google-style docstrings on public functions are
  required by AGENTS.md. Don't flag a real docstring on a real public API just
  because the function is short.
- **Inline argument names for non-obvious booleans**: `func(strict=True,  # raise on unknown keys`)
  is explicitly endorsed by AGENTS.md.
- **`# type: ignore[...]` and similar tool directives** are not comments in
  this sense; don't flag.

## Suggested confidence floor

High confidence when the comment is a near-verbatim English version of the
next line, names a PR/issue/phase/kata, or is a multi-paragraph docstring on
a function whose body is ≤3 lines. Lower confidence on borderline "this
explains a little something" comments — if you cannot articulate what would
be lost by deleting it, suppress.
