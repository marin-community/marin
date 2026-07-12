# Marin lint rules — Complexity lane

Functions, classes, and modules that do too much. This lane is fed advisory static-complexity *leads* (cyclomatic complexity, nesting, size); the leads only point the eye — a finding still requires the rule's intent test, never a raw threshold.

The shared detector harness — audience, detector usage, suppression markers, confidence bands, overlap precedence, output format, and self-evaluation — lives in [`shared.md`](shared.md). Read it first; it governs every lane.

## Using the complexity leads

This lane's prompt is prefixed with a `COMPLEXITY LEADS` block: per-function and
per-class metrics (cyclomatic complexity, max nesting depth, line count, method
count) for the changed files. Treat them as a *map of where to look*, exactly as a
profiler points at hot functions — they are not findings and carry no threshold. A
function at cyclomatic 30 that does one thing is clean; a function at cyclomatic 10
that does four unrelated things is `ml-overloaded-function`. Read the body, apply
the rule's intent, and ignore the number when it disagrees with the code. You may
also flag a unit the leads did not surface, and you must stay silent on a
high-metric unit that is genuinely cohesive.

## Doing too much

### `ml-overloaded-function` — Function body interleaves several distinct responsibilities

**Why it's bad:** A function with an ordinary signature whose body runs several
unrelated jobs in sequence — validation, then two separate DB transactions, then
state/drain transitions — forces the reader to hold every stage at once and leaves
the next editor no seam to change one stage in isolation. This is the body-level
sibling of `ml-monolithic-function`: that rule fires on a multi-mode *signature*
(boolean knobs); this one fires on an ordinary signature wrapping a body that does
too many *distinct, separately-nameable* things. Split into named steps the entry
point composes.

Flag on intent, not length: you must be able to name three or more unrelated
responsibilities at the cited line. A long but genuinely linear routine — one
responsibility, many sequential steps — is not a violation. The complexity leads
(high cyclomatic complexity, deep nesting, large line count) are a hint to look
here, never the trigger. When in doubt, suppress.

**When allowed:** Pre-existing public entry points where splitting would break
callers; a genuinely single-responsibility routine that is merely long.

**Bad example:**
```python
def launch_job(req):
    # validates the request, opens a reservation transaction, opens a second
    # scheduling transaction, runs TPU-shape validation, then walks drain/cancel
    # state transitions — five responsibilities, each a candidate helper.
    ...
```

### `ml-monolithic-function` — Multi-mode function that should be split

**Why it's bad:** One function with three boolean knobs encodes 2³ behaviors
the caller has to reason about. Separate functions compose better and let
callers pick exactly what they need.

**When allowed:** Pre-existing public APIs where splitting would break
callers. New entry points should be narrow.

**Overlap:** when the split is driven by boolean *knobs* in the signature, this is
the rule; when the signature is ordinary but the body does too many distinct
things, use `ml-overloaded-function`.

**Bad example:**
```python
def compute_loss_mask(
    tokens, *, mask_eot: bool, mask_user_turns: bool, mask_assistant_turns: bool
) -> Mask:
    # three orthogonal masks → three functions, composed by the caller.
    ...
```

### `ml-god-class` — Class or module owning several unrelated responsibility clusters

**Why it's bad:** A class whose methods split into clusters that share no state —
RPC dispatch here, DB transactions there, TPU validation, a drain state machine —
is several collaborators wearing one name. Every reader pays for the whole surface
to understand any part, and unrelated changes collide in one file. The same applies
to a module that has become a grab-bag of unrelated top-level members.

Flag on responsibility-mixing, not method or line count: name the distinct clusters
and show they do not share state. A large but cohesive service class — one
responsibility, many methods over shared state — is not a violation. The size leads
(method count, module length, member count) only point the eye.

**When allowed:** Framework-mandated god objects, generated code, or a genuinely
cohesive large class whose methods share state. When the clusters share state,
leave it.

**Bad example:**
```python
class ControllerServiceImpl:  # 45 methods: RPC handlers + DB transactions +
    # TPU validation + a drain state machine — four collaborators in one class.
    ...
```
