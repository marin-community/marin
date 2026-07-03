# Marin lint rules — Meta lane

Holistic, whole-diff rules: smells you can only see by holding the entire change
at once — a concept smeared across many edits, a helper reinvented, a path left
stranded, a payload that wants a shape. Unlike the other lanes, which scan
added/modified hunks for a known *local* shape, this lane reasons over the change
**as a unit**: model what the PR is trying to do, then judge whether the means are
the cleanest path to that end.

The shared detector harness — audience, detector usage, suppression markers,
confidence bands, overlap precedence, output format, and self-evaluation — lives
in [`shared.md`](shared.md). Read it first; it governs every lane. Two things are
specific to this lane:

- **Diff scope is wide; rule scope is narrow.** You may read beyond the diff — the
  whole file, the call graph, a sibling module, an existing helper, a tree-wide
  grep — to confirm a finding (several rules below require it). But you own only
  the `ml-` codes in this file. If you notice a *local* smell (a bad name, one
  overloaded function, a swallowed exception), stay silent — its lane will catch
  it. Fire only where a hunk-scoped pass structurally cannot see the problem: it
  spans files, lives in an unchanged file, or is a property of the whole change.
- **This lane runs only on larger diffs** (>100 changed lines). On small PRs it
  does not run at all, so do not worry about tiny changes here.

These rules are higher-judgment than the local lanes; lean hard on each rule's
confidence floor and its suppressors, and where a rule says so, phrase the finding
as a question to confirm rather than an assertion.

## Aspirational redesign

### `ml-there-must-be-a-better-way` — Whole-PR change wants a cleaner shape (umbrella)

**Why it's bad:** The one judgment a senior reviewer makes that no local rule can:
"I see what you did, but the whole thing wants to be a Protocol / a dataclass / the
helper we already have." Each local rule sees an innocent hunk; only a reader
holding the entire change feels the aggregate friction, and the redesign is
cheapest while the shape is still new. This is the catch-all of last resort: if one
of the sharper rules below fits, emit that instead.

Fire only when ALL hold: (1) you can name ONE concrete, house-idiomatic construct
that replaces the added surface — a named Protocol, a named dataclass, or an
existing helper by qualified name, never an adjective like "cleaner"; (2) the
alternative provably *removes* surface the PR added (cite the lines it deletes),
not just rearranges it; (3) the construct does not cross a layer boundary
(`iris`/`haliax` → `levanter`/`zephyr` → `marin`) and does not collapse a value
that crosses a wire/proto/serialization boundary; (4) confidence ≥0.85. If you
cannot name the construct and the lines it deletes, suppress.

**When allowed:** The added surface is irreducible — three impls that genuinely
share little, a kind-string that is also a persisted/proto value, a positional
shape mandated by JAX/Haliax ergonomics.

**Bad example:**
```python
# handle_memory_sink(), handle_disk_sink(), handle_s3_sink() each take the same six
# args and branch on a kind str threaded through three layers → one Sink Protocol
# with three small impls; the kind-string and the six-arg tuple vanish at the boundary.
```

### `ml-data-clump-threaded` — A parameter group / threaded knob wants one shape

**Why it's bad:** Three fields always passed together are a concept missing its
name; a knob plumbed verbatim through N forwarding layers means every future knob
repeats the N-site edit and each intermediate layer lies about what it needs
("accept only what's necessary"). The local rules see each added param in
isolation; only the whole-diff view sees the same group threaded repeatedly and
that an existing config already rides the path.

Fire on EITHER: (a) the same ≥3-value group in ≥2 new/edited signatures in the same
order and meaning; OR (b) the same new parameter in ≥3 changed signatures where ≥1
intermediate frame only forwards it AND a dataclass/config already travels the full
path (name it). Name the carrier and show it removes construction sites, not just
signature slots — if every site builds the group fresh from distinct sources, a
shared dataclass saves nothing; suppress. Distinct from `ml-config-not-threaded` (a
knob set then *ignored*): here the value reaches its consumer, just in the wrong
shape.

**When allowed:** Numeric axis/shape/coordinate tuples — `(batch, seq, embed)`,
`(start, end, step)`, einsum dims — are intentionally positional in
JAX/Haliax/Pallas code; a dataclass there is ceremony that hurts `jit`/`vmap`
ergonomics. Genuine plumbing where every layer consumes the value.

**Bad example:**
```python
def submit(region, ...): ...        # region: str
def _schedule(region, ...): ...     # only forwards region
def _reserve(region, ...): ...      # only forwards region
def _build_spec(region, ...): ...   # finally reads it
# → put region on the RequestContext already flowing the chain; drop it from 3 signatures.
```

## Diff topology

### `ml-shotgun-surgery` — One concept smeared across many edit sites

**Why it's bad:** The canonical signal of a missing abstraction, and the PR itself
is the evidence: you edited N places to teach the codebase one fact, and the N+1th
place will be missed next time and silently diverge. The author holds the full
model right now and the centralizing move (decorator, dispatch table, dataclass
field, registry) is cheapest before merge. Invisible hunk-by-hunk — each edit looks
trivially correct.

Fire only when the SAME conceptual edit *adds new per-site logic* (a guard, a
branch, a registration call) at ≥4 distinct sites in this diff AND you can name the
concept and the concrete seam that absorbs the change at ZERO call sites. Suppress
mandatory fan-out: a rename, or a new required field on a dataclass that every
construction site must pass — that breadth is correct, not a smell. The proposed
seam must not be a re-export/alias (that would be `ml-compat-shim-not-migration`).
Anchor on the first edited site; list the others in the message.

**When allowed:** Mechanical fan-out that already routes through one source; edits
that merely rhyme without one shared concept; renames and required-param additions.

**Bad example:**
```python
# this PR prepends `if not ctx.authed: raise PermissionError` to 11 RPC handlers
# → one @require_auth decorator (or one check in the dispatch loop): 11 edits → 1.
```

### `ml-echo-across-files` — Cross-file duplication the PR introduces

**Why it's bad:** The local duplication rules are scoped to one function or one
file's pair of sites; they cannot see that file A's new helper and file C's new
helper are the same algorithm born twins in one PR. Copy-paste across modules in a
single change is the most common way real duplication enters a codebase, and it is
exactly invisible to a hunk-scoped pass. Catching it at birth, before both copies
drift, is far cheaper.

Fire only when ≥2 files in this diff carry a block matching on STRUCTURE
(token-identical modulo renames, not "rhymes") of ≥8 logical lines, OR an identical
multi-element literal set (frozenset/tuple) in ≥2 files. Name a concrete shared
home that BOTH sites already depend on AND that already owns this concept's layer —
the extraction must not push higher-layer logic down into a leaf or create a
reverse/cross-layer dependency. Anchor on the first copy; cite the other.

**When allowed:** Two sites in deliberately isolated modules — experiment scripts,
one-off tools, or a **standalone executable script** (e.g. a PEP-723 `uv` script
carrying its own `# /// script` dependency block) that must not import its siblings
— where coupling them is the worse dependency. Same carve-out as
`ml-duplicate-logic-block`. A pure-stdlib helper mirrored into such a script, marked
as a deliberate mirror, is acceptable: there is no shared home both sides can import
without breaking the script's isolation, which is the gate's whole precondition.

**Bad example:**
```python
# ingest/router.py and ingest/worker.py BOTH gain the same 12-line region-allowlist +
# normalize loop, neither pre-existing → extract one normalize_region(raw) in the
# boundary module both already import.
```

### `ml-compat-shim-not-migration` — Back-compat shim left straddling instead of migrating callers

**Why it's bad:** Marin's house rule is explicit and unusually strong — NO BACKWARD
COMPATIBILITY; update all call sites. A shim (an alias, a dual-accepting union, a
`@property` forwarding an old name, "support both for now") is cheaper for the
author and more expensive for everyone after, and it tends to become permanent. The
local `ml-input-type-union` / `ml-speculative-abstraction` rules even *allow* a
one-release adapter — the whole-diff view is what sees the precondition that turns
"allowed adapter" into "should have been a migration": the surviving callers are
all in-repo and few.

Fire only when ALL surviving callers of the old form are inside this repo and
editable, their count is small enough to enumerate (≤~8), and the diff did NOT
already update them — list the call sites. Trigger on a visible alias/shim OR
surviving old-name references the PR left unmigrated. Anchor on the shim/alias line.

**When allowed:** The old form is a wire/serialized/published surface — a config
field read from YAML/checkpoint/`wandb` artifact/proto, or a symbol in a published
`marin-*` package's public API (those *are* published) — or the class participates
in (de)serialization (`from_dict`/`to_dict`/draccus/`asdict`); a caller lives in a
downstream repo; or a comment names a removal trigger and owner (the
AGENTS.md-sanctioned exception). When unsure the old name is purely internal,
suppress.

**Bad example:**
```python
class Worker:
    @property
    def spec(self):                 # renamed spec → task_spec; 6 in-repo callers left
        return self.task_spec       # on `.spec` → rename the 6 callers, delete the property.
```

## Cross-graph reasoning

### `ml-reinvents-existing-helper` — Hand-rolls a helper that already exists in the repo

**Why it's bad:** AGENTS.md's "search the codebase before writing any utility" is
the single most-violated rule by generated code, and it is invisible to every local
lane because the duplicate lives in a file the hunk does not touch. Two
implementations of batching / retry / IO drift independently and the bug surfaces in
prod, not in tests.

Fire only when you can cite the EXISTING symbol by a path you have READ in the
worktree this run and quote its `def` signature line — never rest a finding on a
remembered or assumed symbol; if you cannot open the file and see the def, suppress.
The existing symbol's layer must be import-legal from the new code (`iris`/`haliax`
→ `levanter`/`zephyr` → `marin`): a `marin` helper "reinvented" inside `levanter` is
NOT a finding — importing it would be an illegal reverse dependency, so the local
copy is mandatory. The new code must be behaviorally equivalent (e.g.
trailing-partial-batch handling matches), not merely similar-shaped.

**When allowed:** The existing helper lives behind an optional dependency the new
module cannot assume; no layer-legal import exists; the behaviors differ in a way
that matters.

**Bad example:**
```python
def _chunks(xs, n):
    out = []
    for i in range(0, len(xs), n):
        out.append(xs[i : i + n])
    return out
# levanter/data/utils.py already defines `def batched(seq, n)` → import it (layer-legal here).
```

### `ml-asymmetric-pair` — Paired operation added without its counterpart

**Why it's bad:** Asymmetry is the classic source of leaks and stuck state — a
worker registered but never unregistered, a lock acquired on the happy path but not
released on error, a cache populated by a new writer with no eviction. Not visible
in any single hunk; visible only when you hold the lifecycle as a unit, because the
missing half often belongs in a sibling teardown elsewhere in the change.

Fire only on a recognized symmetric verb pair (closed vocabulary:
register/unregister, open/close, acquire/release, subscribe/unsubscribe,
start/stop, add/remove, serialize/deserialize, incref/decref, enter/exit,
connect/disconnect, lock/unlock). Grep the whole tree for the counterpart verb on
the same receiver and confirm it is absent everywhere, not just in the diff — the
"close" half often lives in a sibling teardown in an unchanged file. Phrase as a
question ("adds `subscribe`; is the `unsubscribe` present on the drain path?").
Anchor on the unpaired "open" call.

**When allowed:** The resource is handed to a context manager / `ExitStack` /
framework / arena that owns teardown; an idempotent fire-and-forget registration
(`atexit`, a signal handler, a metric) with no conventional counterpart; the
counterpart demonstrably already exists outside the diff; a value that legitimately
outlives the function.

**Bad example:**
```python
def subscribe(self, cb):
    self._subscribers.add(cb)
# ...the diff's close()/unsubscribe() never .discard(cb) → add the symmetric removal,
# or hold subscribers behind a context manager / weakref so the pair can't drift.
```

### `ml-orphaned-by-change` — Change strands an old path it should have deleted

**Why it's bad:** `ml-obsolete-after-refactor` only fires when the dead code is
itself in a changed hunk. The expensive case is the opposite: the PR adds the
replacement and updates the callers, and the superseded path sits live in an
UNCHANGED file — every hunk-scoped reviewer passes it. NO-BACKWARD-COMPAT is house
law; a stranded old path is exactly the trap it exists to prevent.

This rule advises *confirmation*, never assertion — phrase the finding as a question
("appears to lose its last static caller; confirm it is not reached via
dispatch/registry/entrypoint before deleting"). Fire only when this diff supplies
the replacement AND you traced that the orphan's last in-repo caller was
removed/rerouted by this diff (cite the caller hunk), AND a whole-tree grep of the
symbol NAME (as a string and as an attribute) returns zero hits beyond the def and
the rerouted callers. Dynamic dispatch is pervasive here — Iris RPC handlers,
Levanter config registries, fsspec protocol dispatch — and is invisible to a static
call graph. Anchor on the orphan's `def` line (in the unchanged file); confidence
≥0.9 — this is the highest-stakes claim in the catalog.

**When allowed:** The orphan is an RPC/handler method, is referenced by name as a
string anywhere, is decorated (`register`/`route`/entrypoint), is exported in
`__init__`/`__all__`, or is a documented migration shim with a removal trigger. Any
dynamic-dispatch indicator → suppress.

**Bad example:**
```python
# this PR routes log delivery through new direct_push() and updates all callers, but
# WorkerProvider.forward_logs() (unchanged file) was reachable only from those callers
# and nothing references it by name → confirm no dispatch path, then delete it.
```
