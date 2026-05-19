# Marin lint rules

Catalog of patterns reviewers in this repo recurrently flag. Each rule has a
short code (`ml-...`), the condition, why it's bad, when it's nevertheless
acceptable, and a bad-pattern example. Rules are *advisory* — surface findings
to the author, never block.

The single source of style truth is [AGENTS.md](../AGENTS.md); rules below cite
the section they enforce. If AGENTS.md changes, the rule inherits the change.

## How to use this file

- **Reviewer / agent**: scan a diff (`git diff main...HEAD -- '*.py'` or the
  smallest equivalent — staged diff, `gh pr diff <n>`, named files). For each
  finding, emit one line in the format described under "Output format" below.
- **Author**: search this file for the code from a finding (`ml-...`) to see
  the rule, why it matters, and when it's OK to ignore.

Only flag added/modified hunks; surrounding context is fair game for judging
intent. Migrations, `__init__.py`, proto definitions, and test fixtures all
count. Anything that looks like an actual bug — file under the closest rule
(usually `defensive`) and let the human decide.

This is not a security review (see `/security-review`), a correctness checker,
or a formatter (ruff / Black already exist; stay out of whitespace, import
order, line length).

---

## Imports

### `ml-local-import` — Use of local imports

**Why it's bad:** A local import is a sign the author didn't properly inspect
the file and introduces maintenance burden — readers can no longer see a
module's dependencies at a glance, and the same import tends to get repeated
inside every function that needs it. Re-inspect the file and lift the import;
refactor if you need to.

**When allowed:** Only to handle external-dependency conditions (a package
only available with a certain extra). The optional-dep case is canonical
and *correct*, not a nit — a `try/except ImportError` or a docstring
noting the extra makes the intent obvious.

Import-cycle workarounds are *not* a stable exception. In well-factored
Python the structural fix always exists: extract a `Protocol` / ABC / shared
dataclass into a third module that both sides depend on, use
`from __future__ import annotations` for type-hint-only references, or use
string forward references in ORM-style declarations. A local import to
break a cycle is *transition debt* — acceptable only mid-refactor, paired
with a comment naming the follow-up issue.

**Bad example:**
```python
def write_chunk(path: str, data: bytes) -> None:
    import zstandard  # zstandard is a hard dep; belongs at module scope

    cctx = zstandard.ZstdCompressor()
    ...
```

### `ml-type-checking-guard` — `TYPE_CHECKING` guard block

**Why it's bad:** AGENTS.md § Code Style forbids `TYPE_CHECKING` guards
outright. They hide real cycles instead of fixing them and split the import
graph across runtime vs. type-check time, which confuses readers and tools.

**When allowed:** Never in new code. Fix the cycle structurally — define a
`Protocol` in the layer that owns the type, and have both sides depend on the
protocol.

**Bad example:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marin.experiments import ExperimentConfig  # fix with a Protocol instead


def run(cfg: "ExperimentConfig") -> None: ...
```

---

## Layering

### `ml-reverse-layer-import` — Reverse-direction import across layers

**Why it's bad:** AGENTS.md § Code Reuse fixes the dependency direction as
`{iris, haliax} → {levanter, zephyr} → marin`. A reverse import (e.g. `from
marin...` inside `lib/iris/`) makes the leaf library un-reusable and creates
a cycle the moment the trunk reaches back.

**When allowed:** Tests inside `lib/<leaf>/tests/` may import from `marin` to
exercise integration paths if marked `@pytest.mark.integration` and the
production code under `src/` stays clean. Top-level tooling under `infra/`,
`scripts/`, and `experiments/` may import any layer.

**Bad example:**
```python
# in lib/iris/src/iris/cluster/something.py
from marin.processing import tokenize  # iris cannot depend on marin
```

### `ml-cross-sibling-import` — Cross-sibling middle-tier import

**Why it's bad:** Not strictly a layering violation, but `lib/levanter/`
importing from `lib/zephyr/` (or vice versa) means a helper both libraries
need has been homed in the wrong place. The right home is the leaf tier
(`iris`/`haliax`) or `marin`, with the dependency pointing one direction.

**When allowed:** Where the cross-import is a deliberate, documented
architectural choice and moving the helper would cost more than it saves.
Surface as a `nit`, not a `warn`.

**Bad example:**
```python
# in lib/levanter/src/levanter/data/foo.py
from zephyr.coordinator import Coordinator  # shared helper, wrong home
```

---

## API shape

### `ml-bool-flag-arg` — Boolean flag selecting between behaviors

**Why it's bad:** Boolean arguments accumulate; they don't extend cleanly to a
third state and they hide intent at the call site (`foo(True, False, True)`).
An enum scales to N states and reads clearly.

**When allowed:** Genuine two-state toggles where the meaning is obvious from
the name and a third state is implausible (e.g. `strict=True` on a parser).

**Bad example:**
```python
def dedupe(rows: list[Row], exact: bool = False) -> list[Row]:
    # second mode lands → second bool. enum DedupMode = {NONE, EXACT, FUZZY} is the right shape.
    ...
```

### `ml-bool-return-status` — `bool` return for a multi-outcome operation

**Why it's bad:** A `bool` return collapses distinct outcomes (success /
timeout / already-flushed) into one bit; callers can't distinguish them and
end up reading the implementation.

**When allowed:** Simple binary predicates (`exists()`, `is_ready()`) and
genuine pass/fail I/O (`write_atomic()` where retry is the only response to
failure).

**Bad example:**
```python
def flush(self, timeout: float) -> bool:
    # callers can't tell "nothing to flush" from "timed out". FlushResult enum is the fix.
    ...
```

### `ml-tuple-return-shape` — Wide tuple return where a dataclass fits

**Why it's bad:** `tuple[dict, str, bool, int]` hides positional semantics;
callers have to count indices and refactors break silently. Three+ fields with
distinct meanings should be a dataclass / NamedTuple.

**When allowed:** Variable-length sequences (`tuple[T, ...]`), 2-tuples where
both elements have obvious roles (key/value pairs), or fixed coordinate-like
tuples.

**Bad example:**
```python
def parse_request(raw: bytes) -> tuple[dict[str, dict], str, bool]:
    # what is each slot? a Parsed dataclass with named fields reads itself.
    ...
```

### `ml-input-type-union` — `X | str` parameter union forcing `isinstance` checks

**Why it's bad:** AGENTS.md § Types — pick one input type. Polymorphic
parameters mean every callee branches on `isinstance` and every caller has to
guess which form is preferred. Normalize once at the boundary.

**When allowed:** Backward-compat adapters that must accept both old and new
calling conventions for one release. New code does not introduce them.

**Bad example:**
```python
def open_dataset(source: Path | str) -> Dataset:
    if isinstance(source, str):
        source = Path(source)
    # normalize once at the public entry point; internal code takes Path.
    ...
```

### `ml-monolithic-function` — Multi-mode function that should be split

**Why it's bad:** One function with three boolean knobs encodes 2³ behaviors
the caller has to reason about. Separate functions compose better and let
callers pick exactly what they need.

**When allowed:** Pre-existing public APIs where splitting would break
callers. New entry points should be narrow.

**Bad example:**
```python
def compute_loss_mask(
    tokens, *, mask_eot: bool, mask_user_turns: bool, mask_assistant_turns: bool
) -> Mask:
    # three orthogonal masks → three functions, composed by the caller.
    ...
```

---

## Types & data structures

### `ml-bare-any` — `Any` where the concrete type is known

**Why it's bad:** AGENTS.md § Types — use `Protocol` for decoupling; avoid
`Any` where the concrete type is known. Bare `Any` defeats the type checker
exactly at the points where it would have caught the next refactor.

**When allowed:** Boundary code that legitimately handles unrelated types
(generic cache value, ad-hoc JSON blob). Document the reason in a brief
comment.

**Bad example:**
```python
def send_entries(payloads: list[Any]) -> None:
    # only ever called with list[logging_pb2.LogEntry]. Type it.
    ...
```

### `ml-non-auto-enum` — Manually numbered enum

**Why it's bad:** Hand-numbered enums (`A = 1; B = 2`) are fragile to reorder
and add nothing over `auto()`. AGENTS.md prefers `enum.auto()`.

**When allowed:** Wire identifiers that must stay stable across versions
(proto enum numbers, serialized IDs). Document that the integer values are
load-bearing.

**Bad example:**
```python
class JobState(Enum):
    PENDING = 1
    RUNNING = 2
    DONE = 3   # use auto() unless these ints cross a wire
```

### `ml-missing-protocol` — Variant-flag dispatch where a Protocol fits

**Why it's bad:** A class that branches on `self.kind == "memory"` vs
`"disk"` is reinventing subclassing badly. Two flavors implementing a
Protocol scale; a growing list of `if` branches doesn't.

**When allowed:** When there really are only two variants and they share
≥80% of their body. Once a third variant appears, refactor.

**Bad example:**
```python
class TreeCache:
    def __init__(self, mode: str):
        self.mode = mode

    def get(self, k):
        if self.mode == "memory": ...
        elif self.mode == "disk": ...
        elif self.mode == "s3":   ...   # third branch → protocol time.
```

### `ml-missing-isinstance-narrow` — Manual type check instead of `isinstance`

**Why it's bad:** Non-`isinstance` type checks (attribute probing, `type(x)
is Foo`) don't narrow under the type checker; pyrefly / mypy can't validate
downstream code.

**When allowed:** Genuinely structural duck-typing where the caller can't
import the concrete type. Rare; prefer a Protocol.

**Bad example:**
```python
if hasattr(payload, "to_proto"):
    payload.to_proto()   # use isinstance(payload, Protoable) and let the checker help.
```

### `ml-raw-dict-vs-dataclass` — Ad-hoc dict for a structured record

**Why it's bad:** AGENTS.md § Types — dataclass/NamedTuple over raw dicts.
Dicts skip schema validation, hide field names from the type checker, and
make evolution painful (rename → silent breakage).

**When allowed:** Truly heterogeneous payloads, JSON deserialization at the
boundary (then convert to a dataclass), or short-lived intermediate state.

**Bad example:**
```python
record = {"id": row.id, "kind": row.kind, "ts": row.ts}
queue.append(record)   # define a @dataclass Record once and reuse.
```

---

## Configuration explicitness

### `ml-env-var-vs-param` — Env var used in place of an explicit parameter

**Why it's bad:** AGENTS.md § Configuration — prefer constructor/config
parameters over env vars. Env vars couple the call to ambient state, can't be
type-checked, and divergent overrides accumulate silently.

**When allowed:** Operational kill-switches and emergency circuit breakers
where the value must flip without a redeploy. The env var should still be
hoisted to a top-level constant and documented.

**Bad example:**
```python
def fetch_mirror_url() -> str:
    # take this as a parameter; only top-level entry points read os.environ.
    return os.environ.get("IRIS_PYPI_MIRROR", "https://pypi.org")
```

### `ml-module-globals` — Module-level mutable state

**Why it's bad:** AGENTS.md § Configuration — composition over inheritance,
explicit parameters over ambient state. Module globals scatter configuration
across the codebase and create order-of-import bugs.

**When allowed:** True constants (frozen sets, immutable lookup tables) at
module scope are fine. The smell is *mutable* globals or globals that hold
runtime-configured state.

### `ml-magic-constant` — Magic string/number repeated without a top-level constant

**Why it's bad:** AGENTS.md § Naming — top-level constants for magic
strings/numbers. Repeated literals drift (one site updated, another not) and
make searches for "where does this value come from" yield nothing.

**When allowed:** A literal that appears exactly once, inside the function
that owns the meaning. Hoist the moment it appears twice.

**Bad example:**
```python
def submit(spec):
    spec["pool"] = "gpu-h100-spot"   # this string appears in 4 files. PoolName.GPU_H100_SPOT.
    spec["timeout"] = 3600
    ...
```

### `ml-config-not-threaded` — Configuration value not threaded to the consumer

**Why it's bad:** A config knob exposed at the deploy/CLI surface but ignored
by the actual consumer (e.g. `cfg.port` set in k8s while the image hard-codes
`--port 10001`) is worse than no knob — operators trust it and get burned.

**When allowed:** Genuinely cosmetic config (logging labels) where the
divergence is harmless. Anything that affects traffic/correctness must
thread.

**Bad example:**
```python
# in deploy config
port: int = 10001

# in entrypoint
def main():
    server.run(port=10001)   # hard-coded; cfg.port is decoration.
```

---

## Defensive code

### `ml-try-except-fallback` — `try/except` fallback instead of fail-fast

**Why it's bad:** AGENTS.md § Error Handling — let exceptions propagate.
Silent fallbacks obscure whether the code is handling a real recoverable
case or papering over a bug.

**When allowed:** Real system boundaries (network, filesystem,
deserialization) where graceful degradation is the documented contract and
the fallback path is tested. Document it.

**Bad example:**
```python
try:
    return json.loads(payload)
except Exception:
    return {}   # caller now can't tell empty-result from parse-failure.
```

### `ml-exception-swallow` — `except Exception` returning `None` / a default

**Why it's bad:** AGENTS.md § Error Handling — never swallow exceptions
unless specifically requested. Returning `None` on parse failure makes the
caller's `if result is None` indistinguishable from "input legitimately
empty."

**When allowed:** Background tasks that must keep running on per-item
failure — but log the exception with context, and emit a metric, never just
swallow.

**Bad example:**
```python
def maybe_parse(s: str) -> dict | None:
    try:
        return json.loads(s)
    except:
        return None   # raise ValueError with context, or take an explicit on_error= flag.
```

### `ml-guard-after-error` — Defensive guard placed after the dereference

**Why it's bad:** An `isinstance(payload, dict)` check inside the
`except` clause is shutting the gate after the horse left — the `.get()`
already ran on the wrong type and produced the exception you're now
catching. Guard at the boundary.

**When allowed:** Never. Move the guard before the dereference.

**Bad example:**
```python
try:
    return payload.get("id")
except AttributeError:
    if isinstance(payload, dict):   # guard belongs above the .get call
        ...
```

## Dead code

### `ml-unused-param` — Unused function parameter

**Why it's bad:** AGENTS.md § Code Style — delete dead code. Unused params
imply a contract that doesn't exist, and they break tools (template
validation, type checkers, callers searching for usages).

**When allowed:** Required by an interface (e.g. a callback signature) —
but then use `_` to make the intent explicit, and only for that case.

**Bad example:**
```python
def render_pvc(image: str, port: int, remote_log_dir: str) -> str:
    # template never references image/port/remote_log_dir; drop them or split per-template.
    return PVC_TEMPLATE.format(...)
```

### `ml-rollout-scaffolding` — Knob added "just for the rollout"

**Why it's bad:** Configuration flags added "to stage safely, removed after
testing" rarely get removed. They accumulate as long-term technical debt and
expand the surface area reviewers must understand.

**When allowed:** Only with an explicit removal trigger in a comment
(`# CRON(2026-06-01)` or "delete after all workers updated to vX.Y") and an
owner. Without those, do not add it.

**Bad example:**
```python
# nervous about this; testing on dev-cluster first, will remove next week.
USE_NEW_RECONCILE = os.environ.get("USE_NEW_RECONCILE") == "1"
```

### `ml-obsolete-after-refactor` — Code obsoleted by an earlier refactor

**Why it's bad:** A helper that handled the old log-forwarding path still
sits in the worker provider after workers started sending logs directly.
Dead branches confuse readers and the next refactor has to figure out
whether they're load-bearing.

**When allowed:** Conditional compatibility shims tied to a known removal
date — comment must name the trigger.

**Bad example:**
```python
def push_logs(worker, entries):
    # workers now send logs directly; this branch is unreachable.
    if worker.legacy_log_path:
        _forward(worker, entries)
```

### `ml-add-then-remove` — Within-branch add-then-remove churn

**Why it's bad:** One commit adds a column / flag / field, a later commit
in the same branch removes it. The intermediate state never deployed, so
the additive change is pure churn — readers have to mentally cancel two
migrations. Just remove the additive change.

**When allowed:** Never. Rebase the addition out.

**Bad example:**
```
migrations/0047_worker_supports_reconcile_rpc.py   # adds column
migrations/0048_drop_worker_supports_reconcile_rpc.py   # drops what 0047 added
```

### `ml-speculative-abstraction` — Abstraction with exactly one implementation

**Why it's bad:** A `Union`, `Protocol`, or generic helper introduced "in
case we add more variants later" costs reader attention now and pays back
only at the second case — by which point the shape is concrete and easy to
refactor anyway.

**When allowed:** When the second variant is in flight, or when this has been explicitly
designed by the user as part of a longer term evolution.

**Bad example:**
```python
TransitionDelta = AttemptMissingOnWorker   # single concrete variant; "widen later"
class WorkerReconcileResultLike(Protocol): ...   # one implementation, ever
db_writes: list[...] = field(default_factory=list)   # always []
```

---

## Duplication

### `ml-duplicate-logic-block` — Same logic block in two+ places

**Why it's bad:** AGENTS.md § Code Reuse — do not create parallel
implementations. Two copies drift: a fix to one is silently absent in the
other.

**When allowed:** Two sites in deliberately isolated modules (experiment
scripts, one-off tools) where coupling them would create worse
dependencies. Three+ copies are never acceptable.

**Bad example:**
```python
def merge_a(rows):
    seen = {}
    for r in rows:
        if r.key in seen: ...
    ...

def merge_b(rows):
    seen = {}
    for r in rows:
        if r.key in seen: ...   # same algorithm; extract _dedupe(rows).
    ...
```

### `ml-parallel-source-impl` — Two production functions doing the same operation

**Why it's bad:** A "legacy translator" sitting next to the new translator
(or `submit_task` next to `enqueue_attempt` differing only in spec source)
is source-cloned production code. Drift here shows up in production, not
in tests.

**When allowed:** During a migration window where both paths are
intentionally live, with a deletion PR linked. See also
`ml-flag-gated-parallel-path` if a flag selects between them.

**Bad example:**
```python
def reconcile_request_from_plan(plan): ...      # new-wire builder
def legacy_translator_request(plan): ...        # old-wire builder from same plan.request.desired
```

### `ml-test-double-mirrors-prod` — Test double re-implements production logic

**Why it's bad:** A fixture or `InProcessFooProvider` that mirrors the
dispatch/translation logic of the SUT passes when the SUT is wrong in the
same way. Test doubles are supposed to isolate, not mirror.

**When allowed:** Recording adapters that observe inputs/outputs without
re-deriving them. Mirroring production logic is never the answer.

**Bad example:**
```python
class InProcessLegacyProvider:
    def reconcile_workers(self, plans):
        # 70 lines mirroring worker_provider._reconcile_one
        ...
```

### `ml-duplicate-test-body` — Copy-pasted test bodies that should be parametrized

**Why it's bad:** Five test functions differing only in input/expected pair
should be one `@pytest.mark.parametrize`. Copying invites the "fix one,
forget the others" failure.

**When allowed:** When the assertions or setup genuinely differ; pytest
parametrize hurts when the bodies are not actually similar.

**Bad example:**
```python
def test_dedupe_empty():
    assert dedupe([]) == []

def test_dedupe_one():
    assert dedupe([1]) == [1]

def test_dedupe_dup():
    assert dedupe([1, 1]) == [1]   # @pytest.mark.parametrize with (input, expected).
```

### `ml-duplicate-constant` — Hardcoded constant duplicated when a canonical source exists

**Why it's bad:** A frozenset of supported regions re-declared in three
modules will drift when a region is added. Derive from one canonical
source.

**When allowed:** Where the apparent "constant" is genuinely two unrelated
sets that happen to share members today.

**Bad example:**
```python
# in a.py
_SUPPORTED_MULTI_REGIONS = frozenset({"us", "eu"})

# in b.py
SUPPORTED_MULTI_REGIONS = frozenset({"us", "eu"})   # import the one in bootstrap.
```

---

## Naming

### `ml-utils-module` — Module named `*_utils.py` / `*_helpers.py`

**Why it's bad:** AGENTS.md § Naming — no `*_utils.py`; use descriptive
names like `text_cleaning.py`. Generic `_utils` modules become dumping
grounds and stop telling readers anything about contents.

**When allowed:** Cross-cutting utilities for a large number of callers
across an entire package — but even then, prefer a descriptive name
(`fs.py`, `time_math.py`).

**Bad example:**
```
lib/marin/src/marin/processing/tokenize/tokenize_utils.py   # rename to byte_pair_encoding.py or similar
```

### `ml-misleading-name` — Name doesn't match what the function returns or does

**Why it's bad:** AGENTS.md § Naming — function names should reflect return
types. `cpu_wall_ms` that measures wall time (not CPU time), or
`labeled_lm_eval` that does generic masked-span eval, mislead future
readers and bugs follow.

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

**Why it's bad:** AGENTS.md § Naming — no abbreviations like `exe`; use
`exec` or full words. Abbreviations save typing once and cost readability
forever.

**When allowed:** Domain-standard short forms (`MAP`/`REDUCE` in enums,
`http`, `url`, `id`).

**Bad example:**
```python
def _list_stg_files(inp_path: str): ...   # staged_files, input_path.
```

### `ml-seconds-suffix` — `_s` suffix for "seconds"

**Why it's bad:** AGENTS.md § Naming — `_s` is reserved-or-banned; seconds
are the assumed unit in this codebase. Naming with `_s` is either
redundant or confusing (`responses_s`? `rows_s`?).

**When allowed:** Never. Use `_ms` / `_us` / `_ns` for non-second units;
plain names for seconds.

**Bad example:**
```python
def wait(timeout_s: float): ...   # timeout: float — seconds are the default.

# or better -- use dedicated domain types
def wait(timeout: Duration)
```

---

## Comments

### `ml-restating-comment` — Comment paraphrases the line below

**Why it's bad:** AGENTS.md § Comments — write comments for subtle logic,
not to restate code. A comment that says what the next line says is pure
noise and rots first.

**When allowed:** Never. If you cannot articulate what would be lost by
deleting the comment, delete it.

**Bad example:**
```python
# Increment the counter
counter += 1
```

### `ml-trivial-docstring` — Docstring narrates a self-evident one-liner

**Why it's bad:** AGENTS.md § Comments — skip docstrings on trivial
functions with clear names. `def get_user_id(user): """Return the user's
id."""` says nothing the signature didn't.

**When allowed:** Public-API functions documented in user-facing reference
docs. Internal one-liners with clear names: no docstring.

**Bad example:**
```python
def get_user_id(user: User) -> str:
    """Return the user's id."""
    return user.id
```

### `ml-multi-paragraph-docstring` — Multi-paragraph docstring on a trivial body

**Why it's bad:** AGENTS.md § Comments — one short line max. Multi-paragraph
docstrings on three-line bodies are an LLM-generated pitfall and bury the
real public-API docs they should have lived alongside.

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

**Why it's bad:** AGENTS.md § LLM-Generated Code Pitfalls — `__all__` is
redundant when the module already exports the names via `from .x import
Foo`, and it drifts (one symbol added, `__all__` not updated).

**When allowed:** Modules that genuinely re-export a subset and want
`from foo import *` to be a narrow set — rare.

**Bad example:**
```python
# __init__.py
from .foo import Foo
from .bar import Bar
__all__ = ["Foo", "Bar"]   # duplicates the imports above; drop it.
```

---

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

---

## Test quality

### `ml-internal-assertion` — Test asserts on internal state instead of observable behavior

**Why it's bad:** AGENTS.md § Testing — prefer integration-style tests that
validate externally-observable behavior. Asserting on imported constants
or private fields fails on refactors that didn't change behavior.

**When allowed:** Where the internal state *is* the contract (e.g. a
serializer's wire format). Otherwise prefer asserting on the output.

**Bad example:**
```python
def test_runner():
    runner = make_runner()
    assert runner._cap == DEFAULT_STEP_CAP   # check the runnable bundle / step config doc instead.
```

### `ml-heavy-mocking` — Test layered with mocks of internal collaborators

**Why it's bad:** Mocking everything but the unit under test ends up
asserting "the SUT calls the mocks the way I said it would" — tautological
and brittle to harmless refactors.

**When allowed:** Mocks at I/O boundaries (network, filesystem, external
services). Internal collaborators: use the real thing or a minimal stub.

**Bad example:**
```python
def test_dispatch(mock_db, mock_log, mock_rpc, mock_clock, mock_metrics):
    # five mocks for a 20-line function. Find the real seam.
    ...
```

### `ml-tautological-test` — Test that fails on implementation change but not on behavior change

**Why it's bad:** AGENTS.md § Testing — tests must fail if behavior is
wrong, not just if implementation changes. Tautological tests train the
team to ignore failures.

**When allowed:** Never. If a test only fires on refactors, delete it or
rewrite it against externally-observable behavior.

**Bad example:**
```python
def test_sort_calls_sort():
    with mock.patch("module.sorted") as m:
        do_work([3, 1, 2])
        m.assert_called_once()   # behavior: did the output come out sorted?
```

### `ml-duck-typed-double` — Hand-rolled test double that drifts from production

**Why it's bad:** A "fake HTTP server that produces the code we told it to
have" or a duck-typed `FakeClient` slowly diverges from the real
implementation. The test passes; the production breaks.

**When allowed:** Generated fakes that derive from the real interface
(e.g. `protoc`-generated stubs) are fine.

**Bad example:**
```python
class FakeIrisClient:
    def submit_task(self, spec, ...):
        # mirrors the real client's normalization; will drift.
        ...
```

### `ml-brittle-string-match` — Test asserts on exact human-readable output

**Why it's bad:** "I might consider removing some of the text matching
tests or making them less specific, otherwise we'll need to update them
with every tweak of the status." Exact-string asserts couple the test to
copy.

**When allowed:** Where the string IS the contract (CLI machine-readable
output, log lines parsed by another tool). Use `assert ... in` or
parse and assert structurally.

**Bad example:**
```python
assert status_line() == "Worker iris-1 (us-central1-a) ready in 12.3s"
# any tweak to copy breaks the test. Assert on the structured fields.
```

### `ml-time-sleep-in-test` — `time.sleep()` in a test body

**Why it's bad:** AGENTS.md § Testing — no `time.sleep()` in tests; inject
`now=time.time()` or mock time. Sleeping races the SUT instead of
controlling it; the test goes flaky under load.

**When allowed:** Genuinely time-bound integration tests (waiting on a
TPU bring-up) marked `@pytest.mark.slow` with a comment naming what the
wait is for.

**Bad example:**
```python
def test_eventual_flush():
    submit_event()
    time.sleep(0.5)   # inject a clock, or poll with a deadline driven by a fake clock.
    assert len(log) == 1
```

### `ml-unittest-class-wrapper` — `class TestFoo(unittest.TestCase)` adding nothing

**Why it's bad:** AGENTS.md § Testing — prefer top-level `def test_*` with
pytest fixtures. A `TestCase` subclass that just groups tests by topic
adds setup ceremony without buying anything.

**When allowed:** Where you genuinely need `setUp/tearDown` semantics that
fixtures can't express (rare).

**Bad example:**
```python
class TestNormalize(unittest.TestCase):
    def test_lower(self):
        self.assertEqual(normalize("Hi"), "hi")
    # delete the class; top-level def test_lower with a fixture.
```

---

## Detector usage

For agents running this catalog against a diff:

### Inputs

Pick the diff that applies, typically the current code versus the merge-base, unless the user explicitly requests a tighter set.

- Feature branch: `git diff main...HEAD -- '*.py'` (triple-dot — diff against the merge base).
- Pre-commit / pre-push: `git diff --cached -- '*.py'`.
- A specific PR: `gh pr diff <number> -- '*.py'`.
- A named file or two: read the file in full.

If `*.py` is empty, emit nothing and stop. If the diff is larger than one pass, drive a second pass file-by-file via `git diff main...HEAD --name-only -- '*.py'`. Do not sample or truncate.

Scan added/modified hunks plus enough surrounding context to judge intent (usually the enclosing function/class). Do not flag pre-existing code in unchanged regions. Migrations, `__init__.py` exports, proto definitions, and test fixtures all count.

Also consult subproject `AGENTS.md` (`lib/iris/AGENTS.md`, `lib/marin/AGENTS.md`, etc.) for files under that subtree; subproject rules override the global ones.

### Severity

- `nit` — by default: imports, naming, comments, docs, duplication.
- `warn` — by default: api-shape, types, dead-code, defensive, config-explicitness, test-quality, layering.

You may upgrade (a name mismatch obscures a thread-safety guarantee) or downgrade (a flagged shape falls under that rule's "when allowed" but you want to surface it for awareness). When you deviate, explain in the message with `(severity: nit→warn because …)`.

There is no `block` severity. Security findings (auth, injection, secrets) belong in `/security-review`.

### Confidence

Every finding has a confidence in `[0.0, 1.0]`:

- `≥0.9` — example is near-verbatim; the reviewer comment writes itself.
- `0.7–0.9` — pattern fits the rule's intent; some context uncertainty.
- `0.5–0.7` — suggestive; emit only for `warn` rules, suppress for `nit`.
- `<0.5` — do not emit.

Do not pad. Empty output is correct. False positives are the failure mode that erodes trust.

### Overlap precedence

Several rules touch adjacent surface.

If a single line legitimately violates two unrelated rules (e.g. `Any` return and a `_v2` suffix), emit two findings.

### Output format

One finding per line. Ruff-compatible:

```
<path>:<line>: <code> [<severity>] (<confidence>) <message>
```

- `<path>` — repo-relative, forward slashes.
- `<line>` — 1-indexed in the file as it exists post-change.
- `<code>` — the `ml-...` code from this file.
- `<severity>` — `nit` | `warn`.
- `<confidence>` — two decimals, e.g. `0.82`.
- `<message>` — ≤200 chars. State the concern; do not propose a fix. If you deviated from default severity, end with `(severity: <reason>)`.

Worked examples:

```
lib/iris/src/iris/cluster/worker/reconcile.py:284: ml-try-except-fallback [warn] (0.90) silent fallback contradicts docstring's MISSING contract; will mask cache bugs
lib/iris/src/iris/cluster/controller/transitions.py:1673: ml-bool-return-status [warn] (0.85) error: str | None encodes two unrelated transactions in one method
lib/marin/src/marin/processing/tokenize/tokenize_utils.py:1: ml-utils-module [nit] (0.70) module name uses generic _utils suffix
lib/iris/src/iris/cluster/worker/task_attempt.py:107: ml-speculative-abstraction [warn] (0.80) sentinel exists only to satisfy one import (severity: nit→warn — survived prior review)
```

If the diff is empty or has no Python files, emit nothing — no "no findings" message, no preamble, no summary, no JSON, no Markdown, no fenced code blocks. One finding per line, no blank lines between them. Do not echo the input.

### Self-evaluation

- **Precision over recall.** A reviewer who sees a false positive once trusts the tool less. When uncertain, suppress.
- **Calibration.** If you wouldn't bet $1 on a finding being valid, score it below 0.7.
- **Stay in scope.** Only the rules in this file. Don't moonlight as a security / perf / style reviewer for things outside the catalog.
- **Anchor in real shapes.** A reader at the cited line should immediately see why you flagged it. If you're reaching, suppress.
- **Severity deviations must be justified** in the message with `(severity: …)`.
