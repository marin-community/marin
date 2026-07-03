# Marin lint rules — Cruft lane

Dead code, duplication, and low-signal tests — code that has stopped earning its place or never did.

The shared detector harness — audience, detector usage, suppression markers, confidence bands, overlap precedence, output format, and self-evaluation — lives in [`shared.md`](shared.md). Read it first; it governs every lane.

## Dead code

### `ml-unused-param` — Unused function parameter

**Why it's bad:** Delete dead code. Unused params imply a contract that
doesn't exist, and they break tools (template validation, type checkers,
callers searching for usages).

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
refactor anyway. The same smell covers a single-caller forwarding wrapper — a
helper invoked from exactly one site that only forwards to one callee, adding a
hop that neither earns a clarifying name nor is reused.

**When allowed:** When the second variant is in flight, or when this has been explicitly
designed by the user as part of a longer term evolution.

**Bad example:**
```python
TransitionDelta = AttemptMissingOnWorker   # single concrete variant; "widen later"
class WorkerReconcileResultLike(Protocol): ...   # one implementation, ever
db_writes: list[...] = field(default_factory=list)   # always []
```

## Duplication

### `ml-duplicate-logic-block` — Same logic block in two+ places

**Why it's bad:** Do not create parallel implementations. Two copies drift:
a fix to one is silently absent in the other.

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

A common shape in this repo: N methods on one class sharing an identical loop +
rate-limiter + cancellation-check + error-log + thread-lifecycle scaffold that
differs only in the inner call they wrap — consolidate into one runner that takes
the inner step as a parameter.

```python
def _run_autoscaler_loop(self): ...   # loop + rate-limit + cancel-check + error-log
def _run_ping_loop(self): ...         # same scaffold, only the wrapped call differs
def _run_checkpoint_loop(self): ...   # → one _run_loop(step, interval) runner.
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

## Test quality

### `ml-slop-test` — "Slop" test: asserts on incidentals, not behavior

**Why it's bad:** The catch-all for low-value tests — they *look* like coverage
but validate nothing real, or pin to incidental detail that breaks on a harmless
edit while real regressions slip through. Pure maintenance burden: every reword,
rename, or refactor forces a test edit that confirms no behavior. Assert on the
externally-observable output or effect instead. If you can't name the production
bug the test would catch, it's slop.

A test is slop when it does any of these:

- **Asserts on a log message's text** — `assert "retrying" in caplog.text`. A log
  line isn't a contract; it gets reworded freely, and whether the code logged at
  all is rarely the behavior under test.
- **Asserts a token appears in a generated command / argv** — `assert "--gpus" in
  cmd`. Tests how the command was assembled, not what it does: a flag rename,
  reorder, or `--port=80` vs `--port 80` breaks it, and a wrong value beside the
  right flag passes it.
- **Asserts on exact human-readable copy** — a status string, formatted error,
  help text. `assert render() == "Worker iris-1 ready in 12.3s"` breaks on any
  copy tweak.
- **Asserts almost nothing** — runs a path then only checks it didn't raise,
  returned non-`None`, or that a mock was called. Coverage without a behavioral
  claim. Includes mocking out the collaborators that do the work, then asserting
  the SUT called the mocks the way you wired them.
- **Is tautological / mirrors the implementation** — re-asserts a literal or
  constant, or recomputes the same expression the code does so it passes by
  construction.
- **Checks what the type checker already guarantees** — `assert isinstance(x,
  Foo)` on a function declared `-> Foo`, `assert hasattr(obj, "field")` on a
  dataclass field, `assert callable(fn)`, `assert MyEnum.A.value == "a"`. pyrefly
  proves these statically; a test of them only fails when the type annotation and
  body already disagree, which the checker catches first.

**When allowed:** When the asserted string or structure genuinely *is* the
contract — machine-readable CLI output, a wire/serialization format, or a log
line a downstream tool parses — and then assert on the *structured field*
(`record.args`, the parsed argv element, the deserialized object), not the
rendered text. A security-critical command flag that can't be exercised
end-to-end may be asserted on the parsed argument, with a comment saying why.

**Bad example:**
```python
def test_retry(caplog):
    do_work()
    assert "retrying after transient error" in caplog.text   # reword breaks it; retry behavior unchecked

def test_build_launch_command():
    cmd = build_launch_command(cfg)
    assert "--gpus" in cmd        # reordering / --gpus=all breaks this; value never checked

def test_process(mock_store):
    process(rows, store=mock_store)
    assert mock_store.write.called   # path ran, but "did it write the right rows?" is untested

def test_make_runner_returns_runner():
    assert isinstance(make_runner(), Runner)   # make_runner is typed -> Runner; pyrefly already proves this
```

### `ml-time-sleep-in-test` — `time.sleep()` in a test body

**Why it's bad:** No `time.sleep()` in tests; inject `now=time.time()` or
mock time. Sleeping races the SUT instead of controlling it; the test goes
flaky under load.

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

**Why it's bad:** Prefer top-level `def test_*` with pytest fixtures. A
`TestCase` subclass that just groups tests by topic adds setup ceremony
without buying anything.

**When allowed:** Where you genuinely need `setUp/tearDown` semantics that
fixtures can't express (rare).

**Bad example:**
```python
class TestNormalize(unittest.TestCase):
    def test_lower(self):
        self.assertEqual(normalize("Hi"), "hi")
    # delete the class; top-level def test_lower with a fixture.
```
