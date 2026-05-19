# api-shape — detector prompt

## AGENTS.md anchor

§ API Design — accept only what's necessary; replace boolean flags with
meaningful parameters; separate classes over boolean flags for variant
behavior. § Types & Data Structures — dataclass/namedtuple over raw dicts;
`StrEnum` over string keys.

## What to look for

Flag interfaces whose *shape* hides intent: boolean flags that select between
behaviors, opaque bool/tuple return values, monolithic parameter sets, and
functions that bake multiple concerns into one entry point. The fix is
usually splitting into separate functions, enum-ifying flags, or returning a
named record. This detector owns "bool→enum" and "tuple→dataclass" findings;
`types` covers `Any`, `auto()`, and narrowing.

## Anchor examples

- **Boolean flag → enum**: "I have almost never created a boolean argument
  without realizing I want to change it to an enum: `enum DedupMode = { NONE,
  EXACT }`." Boolean flags accumulate; enums signal intent and scale to
  three-or-more states.

- **Return bool → enum**: "`Table.flush` and `LogClient.flush` now return a
  `FlushResult` `StrEnum` with `SUCCEEDED` / `TIMEOUT` instead of `bool`...
  callers [now] compare against `FlushResult.SUCCEEDED`." Opaque booleans
  obscure success/timeout distinctions; enums make the return space explicit.

- **Tuple return → dataclass**: "`tuple[dict[str, dict], str, bool]` is fairly
  complicated return type to read, a dataclass would make it easier to
  reason." Complex tuples hide positional semantics; dataclasses name fields
  and type-check cleanly.

- **Config knobs → separate stages**: "In our modern style, these all feel
  like things that should be their own processing stages which can be
  composed, rather than config knobs." Boolean toggles in a function's
  interface force callers to understand all combinations; separate functions
  compose better.

- **Multi-function logic → separate functions**: "This very much so feels
  like it should be split into separate functions which each just compute a
  type of loss mask." One function with multiple code paths for different
  masks couples their logic; separate functions let callers pick exactly
  what they need.

- **Config dataclass → explicit kwargs**: "Dropped the local config dataclass
  and made `download_gh_archive_events(...)` take explicit keyword arguments,
  so the step wrapper now just threads its own params straight through."
  Intermediate config objects add indirection; pass parameters directly when
  they fit.

- **`X | str` parameter union**: AGENTS.md § Types — "Avoid `X | str` unions
  that require `isinstance` checks — pick one input type." Normalize at the
  boundary, not throughout.

## False-positive guidance

- **Acceptable composability**: Functions that return `bool` for simple
  success/failure in I/O (file write, network send) are fine; reserve
  enum-ification for complex state (idle/running/paused/failed) or ambiguous
  outcomes (success vs. timeout).
- **Small dataclass as config**: A 2–3 field frozen `@dataclass` is cleaner
  than five kwargs; the line is fuzzy, but "simple enough to inline"
  suggests kwargs, "semantically grouped" suggests a dataclass.
- **Hidden state is necessary**: When a boolean genuinely encodes two
  indivisible behaviors (e.g., `streaming=True` disables buffering
  everywhere), document it clearly; if it's scattered across the function,
  split instead.
- **Legacy APIs and adapters**: Backward-compat shims that accept both old
  and new calling conventions are acceptable; flag new code that introduces
  such polymorphism.
- **Protocol-level kwargs**: Accept `**kwargs` in protocol methods if the
  base defines a minimal set and subclasses extend gracefully (e.g.,
  provider plugins).

## Suggested confidence floor

Flag code that uses boolean arguments or returns opaque bool/tuple without
clear semantic names. Enum-ification and dataclass refactors are reliably
valuable in this codebase; treat matches high-confidence unless the code is
a thin adapter layer or a deliberate legacy shim.
