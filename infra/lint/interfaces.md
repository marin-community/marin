# Marin lint rules — Interfaces lane

Imports, layering, public API shape, and type/data-structure choices — how modules depend on each other and how functions and types present themselves.

The shared detector harness — audience, detector usage, suppression markers, confidence bands, overlap precedence, output format, and self-evaluation — lives in [`shared.md`](shared.md). Read it first; it governs every lane.

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
noting the extra makes the intent obvious. Mark such an import with
`# noqa: PLC0415` on the `import` line to record the exception explicitly;
see "Suppression markers" under "Detector usage".

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

**Why it's bad:** `TYPE_CHECKING` guards are forbidden outright. They hide
real cycles instead of fixing them and split the import graph across runtime
vs. type-check time, which confuses readers and tools.

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

## Layering

### `ml-reverse-layer-import` — Reverse-direction import across layers

**Why it's bad:** The dependency direction in this repo is
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

**Bad example:**
```python
# in lib/levanter/src/levanter/data/foo.py
from zephyr.coordinator import Coordinator  # shared helper, wrong home
```

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

**Why it's bad:** Pick one input type. Polymorphic parameters mean every
callee branches on `isinstance` and every caller has to guess which form is
preferred. Normalize once at the boundary.

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

## Types & data structures

### `ml-bare-any` — `Any` where the concrete type is known

**Why it's bad:** Bare `Any` defeats the type checker exactly at the points
where it would have caught the next refactor. Use a `Protocol` or the concrete
type instead.

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
and add nothing over `auto()`. Prefer `enum.auto()` unless the integer values
cross a wire.

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
is Foo`) don't narrow under the type checker; pyrefly can't validate
downstream code.

**When allowed:** Genuinely structural duck-typing where the caller can't
import the concrete type. Rare; prefer a Protocol.

**Bad example:**
```python
if hasattr(payload, "to_proto"):
    payload.to_proto()   # use isinstance(payload, Protoable) and let the checker help.
```

### `ml-raw-dict-vs-dataclass` — Ad-hoc dict for a structured record

**Why it's bad:** Prefer dataclass/NamedTuple over raw dicts. Dicts skip
schema validation, hide field names from the type checker, and make evolution
painful (rename → silent breakage).

**When allowed:** Truly heterogeneous payloads, JSON deserialization at the
boundary (then convert to a dataclass), or short-lived intermediate state.

**Bad example:**
```python
record = {"id": row.id, "kind": row.kind, "ts": row.ts}
queue.append(record)   # define a @dataclass Record once and reuse.
```
