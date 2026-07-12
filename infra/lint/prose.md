# Marin lint rules — Prose lane

Naming, comments, and documentation — how the code names and describes itself.

The shared detector harness — audience, detector usage, suppression markers, confidence bands, overlap precedence, output format, and self-evaluation — lives in [`shared.md`](shared.md). Read it first; it governs every lane.

## Naming

### `ml-utils-module` — Module named `*_utils.py` / `*_helpers.py`

**Why it's bad:** No `*_utils.py`; use descriptive names like
`text_cleaning.py`. Generic `_utils` modules become dumping grounds and stop
telling readers anything about contents.

**When allowed:** Cross-cutting utilities for a large number of callers
across an entire package — but even then, prefer a descriptive name
(`fs.py`, `time_math.py`).

**Bad example:**
```
lib/marin/src/marin/processing/tokenize/tokenize_utils.py   # rename to byte_pair_encoding.py or similar
```

### `ml-misleading-name` — Name doesn't match what the function returns or does

**Why it's bad:** Function names should reflect return types and behavior.
`cpu_wall_ms` that measures wall time (not CPU time), or `labeled_lm_eval`
that does generic masked-span eval, mislead future readers and bugs follow.

**When allowed:** Never knowingly. Rename when discovered.

**Bad example:**
```python
cpu_wall_ms = task_wall_time - start_time   # not CPU time. task_wall_ms.

def labeled_lm_eval(model, data):   # generic masked-span eval. masked_span_eval.
    ...
```

### `ml-vestigial-qualifier` — Qualifier with no surviving contrast

**Why it's bad:** `reconcile_workers_via_reconcile`, `_v2`, `_new`,
`_legacy`, `_compat` all imply two variants when there is one. They
propagate (callers copy the name) and the contrast they referred to is
already gone.

**When allowed:** The qualifier still disambiguates because the contrasting
variant still exists *and is not flag-gated for removal* — file the
flag-gated case under `ml-flag-gated-parallel-path` instead.

**Bad example:**
```python
def reconcile_workers_via_reconcile(...): ...   # stutter; just reconcile_workers.
attempt_id_compat: str                          # "compat" with what?
```

### `ml-abbreviated-name` — Cryptic abbreviation in a name

**Why it's bad:** No abbreviations like `exe`; use `exec` or full words.
Abbreviations save typing once and cost readability forever.

**When allowed:** Domain-standard short forms (`MAP`/`REDUCE` in enums,
`http`, `url`, `id`).

**Bad example:**
```python
def _list_stg_files(inp_path: str): ...   # staged_files, input_path.
```

### `ml-seconds-suffix` — `_s` suffix for "seconds"

**Why it's bad:** Seconds are the assumed unit in this codebase, so the `_s`
suffix is either redundant or confusing (`responses_s`? `rows_s`?).

**When allowed:** Never. Use `_ms` / `_us` / `_ns` for non-second units;
plain names for seconds.

**Bad example:**
```python
def wait(timeout_s: float): ...   # timeout: float — seconds are the default.

# or better -- use dedicated domain types
def wait(timeout: Duration)
```

## Comments

### `ml-restating-comment` — Comment paraphrases the line below

**Why it's bad:** Comments are for subtle logic, not for restating code. A
comment that says what the next line says is pure noise and rots first.

**When allowed:** Never. If you cannot articulate what would be lost by
deleting the comment, delete it.

**Bad example:**
```python
# Increment the counter
counter += 1
```

### `ml-trivial-docstring` — Docstring narrates a self-evident one-liner

**Why it's bad:** Skip docstrings on trivial functions with clear names.
`def get_user_id(user): """Return the user's id."""` says nothing the
signature didn't.

**When allowed:** Public-API functions documented in user-facing reference
docs. Internal one-liners with clear names: no docstring.

**Bad example:**
```python
def get_user_id(user: User) -> str:
    """Return the user's id."""
    return user.id
```

### `ml-multi-paragraph-docstring` — Multi-paragraph docstring on a trivial body

**Why it's bad:** Internal docstrings should be one short line max.
Multi-paragraph docstrings on three-line bodies are an LLM-generated pitfall
and bury the real public-API docs they should have lived alongside.

**When allowed:** Genuinely complex public APIs where Google-style sections
(`Args:`, `Returns:`, `Raises:`) document non-obvious contracts.

**Bad example:**
```python
def normalize(s: str) -> str:
    """
    Normalize a string.

    This function takes a string and returns its normalized form.
    The normalization process consists of stripping whitespace ...
    """
    return s.strip().lower()
```

### `ml-impl-narration-docstring` — Docstring narrates implementation, not the caller contract

**Why it's bad:** A docstring states what a function does *for its caller* — the
contract. Narrating *how* it is implemented is noise to the caller and rots when the
implementation changes. The tells: a sentence describing the mechanism ("a single
atomic `UPDATE ... RETURNING`, so no TOCTOU window"), self-congratulation ("single
source of truth for the proto fields"), caller-enumeration ("Used by cancel_job and
prune"), or a cross-reference to a sibling function ("same fallback order as the
ListEndpoints branch") — git grep finds callers, and the cross-ref goes stale. If an
implementation detail is genuinely load-bearing — a non-obvious correctness invariant,
or a performance choice a later edit would naively undo — put it in an inline comment
*next to the code it explains*, not in the docstring. A property/getter that only
returns a field is the same smell in method form: expose the attribute.

**When allowed:** When the detail *is* the contract the caller must know — a nullable
parameter's meaning, an edge-case return (`NULL str_value` → `""`), an ordering or
idempotency guarantee, or what the caller must pass. State the behavior, not the
mechanism.

**Bad example:**
```python
def revoke_login_keys(db, user_id, now) -> list[str]:
    """Revoke a user's login keys. Returns the revoked key_ids.

    Single atomic ``UPDATE ... RETURNING`` so the revoke and the returned ids come
    from one statement — no read-then-write TOCTOU window.   # how, not what — delete it
    """
```

### `ml-pr-reference-comment` — Comment names a task / PR / kata / phase

**Why it's bad:** "Added for the canary ferry flow (see PR #5712)" belongs
in the PR description and git blame. In source it rots — a reader six
months later cannot recover the context and the reference becomes
misleading.

**When allowed:** A permanent URL or ADR path that the comment is *linking
to*, not a transient kata short-code or sprint name.

**Bad example:**
```python
# Added for the canary ferry flow (see PR #5712)
retry_attempts = 3
```

### `ml-bare-todo` — `TODO` without owner or trigger

**Why it's bad:** Bare TODOs accumulate. An actionable TODO names the
trigger ("after the migration lands") or the owner; without one it's a
note signalling work without enabling it.

**When allowed:** TODOs in throwaway experiment scripts. Production code:
name the trigger.

**Bad example:**
```python
# TODO: clean this up
```

### `ml-init-all-export` — `__all__` listing every public symbol in `__init__.py`

**Why it's bad:** `__all__` is redundant when the module already exports the
names via `from .x import Foo`, and it drifts (one symbol added, `__all__`
not updated).

**When allowed:** Modules that genuinely re-export a subset and want
`from foo import *` to be a narrow set — rare.

**Bad example:**
```python
# __init__.py
from .foo import Foo
from .bar import Bar
__all__ = ["Foo", "Bar"]   # duplicates the imports above; drop it.
```

## Documentation

### `ml-stale-docstring` — Docstring describes the old behavior

**Why it's bad:** Readers trust docstrings. Stale parameter descriptions
or "returns True if X" lines that no longer match the implementation cause
callers to read the source — and the next refactor misses the docstring.

**When allowed:** Never knowingly. If you discover one, update it.

**Bad example:**
```python
def find_groups(events):
    """Return True if the detector labels a group HOSTILE."""
    # implementation no longer returns False merely on HOSTILE label.
    ...
```

### `ml-undocumented-return` — Non-obvious return value with no docstring

**Why it's bad:** A function returning `bool` or `int | None` where the
semantics aren't clear from the name forces callers to read the body.

**When allowed:** Names that already convey the return (`is_ready`,
`count`) — no docstring needed.

**Bad example:**
```python
def flush(self) -> bool:
    # what does True mean — flushed? already-empty? timed-out? Say so.
    ...
```

### `ml-stale-inline-comment` — Inline comment describes a previous version

**Why it's bad:** A comment that once said "word-level shingling" remains
after the code switched to character-level. Comments help readers reason;
outdated ones mislead about intent.

**When allowed:** Never. Update or delete.

**Bad example:**
```python
# do word-level shingling
shingles = [s[i : i + k] for i in range(len(s) - k + 1)]   # actually character-level
```

### `ml-docstring-contradicts-impl` — Docstring promises behavior the code doesn't deliver

**Why it's bad:** A `run_corpus_mode()` documented "read-only" but opening
a live `DuckDBLogStore` is not passive — callers relying on the contract
get data corruption or surprise side effects. This is the most expensive
form of stale docs.

**When allowed:** Never.

**Bad example:**
```python
def run_corpus_mode(path: str) -> None:
    """Read-only: inspect the corpus without mutating state."""
    store = DuckDBLogStore.open(path, mode="rw")   # opens for write.
    ...
```

### `ml-rotting-historical-ref` — Source reference to a rollout phase / kata / migration

**Why it's bad:** "Phase B+", "kata h9r9", "see migration 0047 which added
this column" are scaffolding vocabulary that mean nothing once the work
lands. Six months later the comment is actively misleading.

**When allowed:** Durable identifiers — a stable issue URL, an ADR path,
a module-level invariant. Rolling project vocabulary, never.

**Bad example:**
```python
# Phase C: re-enabled after kata h9r9 unblocked
timeout = 60
```
