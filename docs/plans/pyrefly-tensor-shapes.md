# Pyrefly tensor/array shape typing: investigation and recommendation

**Status:** Implemented (config change landed alongside this report). Part of the
Pyrefly coverage ratchet, issue #6219.

**TL;DR.** Pyrefly 1.0.0 ships an experimental native shape-typing system that
*also understands jaxtyping syntax*. Turning it on with a single config line —
`tensor-shapes = true` — makes Pyrefly parse the dimension tokens inside
`Float[Array, "B N"]` as a shape specification instead of as forward-reference
type names. That eliminates **all 141** jaxtyping `unknown-name` false positives
with **zero collateral** on any other error code, which in turn lets us drop the
blanket `unknown-name = false` suppression and enable the code for real. We do
**not** get static shape *checking* of our `jax.Array` / haliax `NamedArray`
code out of this (Pyrefly's shape inference is PyTorch-oriented and does not model
`jax.Array`), and we did not expect to — jaxtyping's shape checks are runtime-only
for every type checker. The recommendation is **option (a): adopt the native
`tensor-shapes` support.**

A separate question — *can we replace runtime-only jaxtyping with
Pyrefly-**checked** shapes to gain real static shape verification?* — was spiked
and answered **no, not practically, today**; see the Feasibility section below.

---

## Why this matters

On Pyrefly the `unknown-name` error code was disabled because a full-project run
produces **144** errors under it. Of those, **141 are false positives**: Pyrefly
reads the dimension tokens inside jaxtyping string-shape annotations
(`Float[Array, "B N"]`, `Int[Array, "TK D"]`, `Bool[Array, "E"]`) as if `B`, `N`,
`TK`, `E`, `T`, `groups`, `heads` were forward-reference *names* and reports
`Could not find name \`B\``. They live almost entirely in
`lib/levanter/src/levanter/grug/_moe/**` and `lib/levanter/src/levanter/kernels/**`,
plus a few in `eval.py`. Only **3 are real bugs**.

The blanket suppression (`unknown-name = false`) is the worst of both worlds: it
silences the 141 false positives *and* hides the 3 real undefined-name bugs. The
goal was to find a path that removes the false positives **without** giving up the
real checking.

## Two shape systems in play

1. **jaxtyping string-DSL** — `Float[Array, "batch seq"]`. Used heavily in
   levanter grug/kernels. This is what trips `unknown-name`.
2. **haliax `NamedArray` + `Axis`** ([`lib/haliax/src/haliax/`](../../lib/haliax/src/haliax/core.py)) —
   a named-axis array. haliax layers its own jaxtyping-compatible DSL on top via
   [`haliax.haxtyping`](../../lib/haliax/src/haliax/haxtyping.py)
   (`ht.Float[NamedArray, "batch embed ..."]`). See
   [haliax's typing docs](../../lib/haliax/docs/typing.md).

## Root cause (reproduced)

jaxtyping is explicitly designed to be a **no-op for static type checkers**. Under
`typing.TYPE_CHECKING`, `jaxtyping/__init__.py` re-exports its dtype classes from
`jaxtyping/_indirection.py`, which is literally:

```python
from typing import (
    Annotated as Bool,
    Annotated as Float,
    Annotated as Int,
    Annotated as Shaped,
    ...,
)
```

So statically `Float[Array, "B N"]` is `Annotated[Array, "B N"]`: per PEP 593 the
`"B N"` is **metadata**, not a type, and the annotation reduces to just `Array`.
The jaxtyping FAQ states this directly: *"An annotation of the form
`dtype[array, shape]` should be treated as just `array` by a static type checker."*
mypy and pyright honour this; the shape string is inert.

Pyrefly nonetheless reports `unknown-name`. The reason is **our** configuration,
not a jaxtyping bug. `pyproject.toml` sets `skip-interpreter-query = true` (so
Pyrefly does not pull workspace members in from site-packages — deliberate, see
the coverage-ratchet notes). A side effect is that the third-party `jaxtyping`
package is **never resolved**: `Float` is an unresolved import, so Pyrefly cannot
see the `Annotated` alias. Faced with `Unknown[Array, "B N"]` in an annotation
position, it falls back to generic-subscription semantics, where a string argument
is a **forward reference** — and it tries to resolve the name.

This is confirmed precisely by probing which shape strings fire (Pyrefly 1.0.0):

| Annotation | Reported? | Why |
|---|---|---|
| `Float[Array, "E"]` | `unknown-name E` | `E` parses as a Name expression → lookup fails |
| `Float[Array, "batch"]` | `unknown-name batch` | single identifier |
| `Float[Array, "heads"]` | `unknown-name heads` | single identifier |
| `Float[Array, "B*N"]` | `unknown-name B` **and** `N` | parses as the expression `B*N` |
| `Float[Array, "B N"]` | — | `B N` is a syntax error as an expression → skipped |
| `Float[Array, "*batch"]` | — | starred expression → skipped |
| `Float[Array, "4 5"]` | — | not a valid expression → skipped |
| `Float[Array, ""]` | — | empty → skipped |

The direct confirmation that the mechanism is forward-reference evaluation:
`Annotated[int, "E"]` (no jaxtyping involved) produces **0 errors** — Pyrefly
correctly treats the metadata string as a string. The bug is only that an
*unresolved* `Float` is not seen as `Annotated`.

### The 144 errors, classified

Inventory from a full-project run with `unknown-name` temporarily enabled:

- **141 false positives** — dimension tokens inside jaxtyping annotations:
  `B`×99, `N`×9, `TK`×8, `groups`×7, `E`×6, `heads`×4, `T`×3, plus `vocab`,
  `tag`, `label`, `EL`, `BB` (all shape tokens that merely *look* like names).
  Files: `lib/levanter/src/levanter/kernels/pallas/**` and
  `lib/levanter/src/levanter/grug/**`, plus `eval.py`.
- **3 real bugs** — genuine undefined names, **not** inside jaxtyping annotations:
  - `lib/levanter/src/levanter/eval_harness.py:572` — `os.makedirs(...)` with no
    `import os`.
  - `lib/levanter/src/levanter/trainer.py:596` and `:617` — `callbacks.profile(...)`
    / `callbacks.compute_validation_loss(...)` where the bare name `callbacks` is
    never imported (the module imports `from levanter.callbacks import ...`, not
    the submodule as `callbacks`; likely fallout from #5594 hoisting lazy imports).

These 3 are real and are owned by the wave-1 bug-fix sessions under #6219; they are
tracked in `.pyrefly-baseline.json` here (see below), not fixed in this change.

## Options considered

### (a) Adopt Pyrefly's native `tensor-shapes` support — **recommended**

Pyrefly 1.0.0 has an experimental native shape-typing system
([docs](https://pyrefly.org/en/docs/tensor-shapes/),
[configuration](https://pyrefly.org/en/docs/configuration/)). It is enabled by the
config key `tensor-shapes` (default `false`; CLI `--tensor-shapes true`) and,
critically, it **accepts jaxtyping syntax** as a front-end — `Float[Tensor, "batch
channels"]` is parsed into its shape system rather than as a forward reference.
This was added in Pyrefly issues
[#925](https://github.com/facebook/pyrefly/issues/925) and
[#743](https://github.com/facebook/pyrefly/issues/743). It builds on PEP 646
(`TypeVarTuple`), which Pyrefly supports.

Measured effect on our tree (full-project run, `unknown-name` enabled):

| Error code | `tensor-shapes` off | `tensor-shapes` on |
|---|---|---|
| `unknown-name` | **144** | **3** |
| `bad-specialization` | 65 | 65 |
| `bad-return` | 33 | 33 |
| every other code | *unchanged* | *unchanged* |

The two runs differ in **exactly one code**: `unknown-name`, 144 → 3. All 141
false positives are gone; the only residue is the 3 real bugs. No other diagnostic
moves. Confirmed to work as a `pyproject.toml` config key (not just the CLI flag),
so the pinned pre-commit invocation (`uvx pyrefly@1.0.0 check --baseline ...`)
picks it up with no change to `infra/pre-commit.py`.

**What it does not do:** it does not add real static shape checking for our arrays.
Pyrefly's shape inference is PyTorch-`Tensor`-oriented; our arrays are `jax.Array`
(jaxtyping's `Array`) and haliax `NamedArray`. A deliberate dimension mismatch —
`def f(x: Float[Array, "A B"]) -> Float[Array, "A D"]: return x` — is **not**
flagged. With `tensor-shapes` on, Pyrefly parses the jaxtyping strings (which
removes the forward-reference false positive) but does not model `jax.Array`
shapes, so it neither verifies nor mis-verifies them. That is the same static
behaviour every other type checker gives for jaxtyping; the actual shape checks
stay where they have always been — at runtime, via jaxtyping's import hook /
typeguard. So this is **false-positive elimination, not new shape verification** —
which is exactly what was achievable.

Caveat: the feature is marked experimental (*"API and behavior may change without
notice"*). This is mitigated by pinning Pyrefly (currently `1.0.0` in
`infra/pre-commit.py`); version bumps are deliberate and regenerate the baseline,
so a future behavioural change would be caught at that gate, not silently.

### (b) Keep jaxtyping, fix the false positives with a stub — rejected

A hand-written `jaxtyping.pyi` on Pyrefly's `search-path` that mirrors
`_indirection.py` (`from typing import Annotated as Float`, etc.) also eliminates
the false positives — *if* it uses the import-alias form. The assignment form
`Float = Annotated` does **not** work (jaxtyping's own comment warns of this:
`Annotated` is a typeform and cannot be assigned; reproduced — it leaves the false
positives in place).

The problem: shipping a stub makes Pyrefly **resolve** `jaxtyping`, and an
*incomplete* stub then turns the (suppressed) `missing-import` into **41 new
`missing-module-attribute` errors** — an *enabled* code — for every jaxtyping name
the real code imports that the stub omits (`Float32`, `Int8`, `jaxtyped`,
`PRNGKeyArray`, …). Measured: `missing-module-attribute` jumps 1 → 42. To be safe
the stub would have to mirror jaxtyping's **entire** public API and be kept in sync
with it. That is strictly more surface area and maintenance than a one-line config
key, for the identical outcome (no real shape checking either). Rejected.

### (c) Status-quo blanket suppression (`unknown-name = false`) — rejected

This is what we are replacing. It hides the 3 real bugs and gives up an otherwise
useful error code. Strictly worse than (a).

### haliax `NamedArray` static shape checking — out of scope today

`NamedArray` ([`core.py`](../../lib/haliax/src/haliax/core.py)) is a plain class
(`class NamedArray(metaclass=NamedArrayMeta)`); it is **not** generic over its axes
or shape, so Pyrefly has no static handle on the named-axis structure — it sees an
opaque `NamedArray`. `haliax.haxtyping` uses the same `Annotated`-alias trick as
jaxtyping, so `tensor-shapes = true` neutralises haliax shape strings the same way
(none of the 144 false positives were in haliax files in the first place — the
heavy direct-`jaxtyping` usage is all in levanter). Expressing named-axis shapes
for *real* static checking would need either a redesign of `NamedArray` around
PEP 646 variadic generics keyed by axis identity (not how haliax is built) or a
bespoke Pyrefly plugin. Neither is justified by the current pain and both are well
beyond this task.

## Recommendation

**Adopt option (a).** Two-line `pyproject.toml` change, implemented in this branch:

1. Add `tensor-shapes = true` to `[tool.pyrefly]`.
2. Flip `unknown-name = false` → `unknown-name = true` in `[tool.pyrefly.errors]`.
3. Record the 3 real bugs in `.pyrefly-baseline.json` (3 new entries) so the gate
   passes while they are fixed under the wave-1 work. When those land, a baseline
   regen drops the entries.

This is a strict improvement over the status quo: it removes the 141 false
positives *and* re-enables `unknown-name`, so genuinely undefined names are caught
going forward (the 3 existing ones become visible/tracked instead of hidden).

**Effort:** ~1 hour including this investigation; the change itself is two config
lines plus three baseline entries. **Payoff:** 141 false positives eliminated, one
real error code re-enabled, 3 latent bugs surfaced. No new file to maintain, no
dependency on an incomplete stub.

Verified green end-to-end with the required entry point:
`./infra/pre-commit.py --all-files` (the Pyrefly stage passes at the pinned
`pyrefly@1.0.0`).

## Feasibility: replace runtime-only jaxtyping with Pyrefly-*checked* shapes?

A natural follow-up: our jaxtyping annotations are checked only at runtime
(via jaxtyping's import hook / typeguard) and are invisible to Pyrefly. Could we
instead express shapes in Pyrefly's native system and get **real static** shape
checking — catching a transpose or rank bug at type-check time? This was spiked
directly. **Verdict: technically possible in narrow forms, but not a practical
replacement for our `jax.Array` / haliax `NamedArray` code today.** Keep jaxtyping
for runtime checks; adopt `tensor-shapes = true` only to remove the static false
positives.

### What the spike established

- **Pyrefly's tensor recognition is structural, not hard-coded to `torch`.** A
  class becomes shape-bearing by being decorated `@shaped_array(shape="Shape")`
  (imported from a `shape_extensions` stub) over a variadic `class Foo[*Shape]`.
  The recognizer keys off that decorator + module name, never the qualified name
  `torch.Tensor` (`pyrefly/lib/export/special.rs`,
  `crates/pyrefly_types/src/shaped_array.rs`).
- **Positional shape checking does *not* need the experimental feature at all.** A
  plain `class Arr(Generic[*Shape])` (`TypeVarTuple`) already makes Pyrefly flag a
  transpose (`Arr[B, S]` returned as `Arr[S, B]`) and a rank mismatch
  (`Arr[B, S]` → `Arr[B]`) as ordinary generic-variance errors — with
  `tensor-shapes` **off**. The `tensor-shapes` feature's genuine value-add over
  vanilla generics is (a) parsing the jaxtyping *string* DSL and (b) `Dim`
  integer-literal arithmetic (`Dim[3] * Dim[4] -> Dim[12]`).

### The blockers for our codebase

1. **The jaxtyping *string* path (our actual annotation style) does not accept a
   non-torch array.** Every attempt to register a custom `@shaped_array` class as
   the array argument of `Float[Array, "B S"]` failed with
   `First argument to jaxtyping annotation must be a tensor class`. The full torch
   fixture stub that makes `Float[Tensor, "batch seq"]` check is **not in
   Pyrefly's public tree** (it lives in a Meta-internal `test/.../fixtures/torch/`
   path). So we cannot keep our existing `Float[Array, "B N"]` annotations *and*
   have Pyrefly check them — the string front-end is currently wired for torch.
2. **Getting any checking therefore means abandoning the jaxtyping string DSL**
   and rewriting to native `Array[B, S]` generics — ~**944** annotations across
   ~**27** levanter files (`grug/`, `kernels/`), and either dropping jaxtyping's
   runtime checks or carrying both systems.
3. **Python 3.12 requirement.** The ergonomic native form `def f[B, S](x: Array[B, S])`
   uses PEP 695 type-parameter lists, which Pyrefly rejects on 3.11
   (`Cannot use type parameter lists on Python 3.11`). This repo is
   `requires-python = ">=3.11"` and targets py310/py311 throughout. The pre-3.12
   `B = TypeVar("B")` spelling works but is verbose at 944-annotation scale.
4. **Stubbing out `jax.Array` is invasive.** Making `jax.Array` shape-bearing
   means shipping a stub that *replaces* jax's real `Array` with
   `@shaped_array class Array[*Shape]`, discarding its large real API — which
   reproduces (and amplifies) the incomplete-stub problem documented in option (b)
   above: a flood of new `missing-attribute` / `bad-argument` errors. The clean
   fix would be jax/jaxtyping shipping Pyrefly-compatible shaped stubs upstream.
5. **haliax `NamedArray` cannot be expressed at all.** Its axes are *named*, not
   positional; a `[*Shape]` variadic of positional dims has no place to carry axis
   identity. This would need a redesign or a bespoke Pyrefly plugin.
6. **Per-function only.** Pyrefly's own docs note the jaxtyping front-end has "no
   way to share symbolic dimensions across variables and functions in a class …
   This limits jaxtyping to individual functions." End-to-end shape flow through a
   model — where most real shape bugs live — is not covered.
7. **Experimental, with arithmetic gaps.** Besides the "may change without notice"
   disclaimer, the reference docs list `int * Dim -> Unknown` and
   `N * (X // N)` not simplifying to `X` — exactly the reshape/flatten arithmetic
   our kernels are full of, each needing a `type: ignore`.

### Bottom line

A serious multi-week effort with upstream dependencies (Pyrefly accepting
non-torch arrays in the jaxtyping path; ideally jax/jaxtyping shipping shaped
stubs), gated on a py3.12 move, yielding only per-function checking with
arithmetic gaps and nothing for `NamedArray`. The payoff is marginal over the
runtime checks jaxtyping already gives us. **Recommendation: do not replace
jaxtyping now.** Revisit if/when (a) the repo moves to py3.12, (b) `tensor-shapes`
leaves experimental and its jaxtyping front-end officially supports JAX/NumPy
arrays (track [pyrefly #2332](https://github.com/facebook/pyrefly/issues/2332)),
and (c) jax/jaxtyping ship Pyrefly-compatible shaped stubs. The kernels — where
shapes are simplest and most local — would be the place to pilot it then.

## Reproduction notes

```bash
# Minimal repro of the false positive (Pyrefly 1.0.0, jaxtyping unresolved):
printf 'from jaxtyping import Float, Array\nx: Float[Array, "E"]\n' > /tmp/x.py
uvx pyrefly@1.0.0 check /tmp/x.py        # -> Could not find name `E` [unknown-name]

# The fix, in isolation:
uvx pyrefly@1.0.0 check /tmp/x.py --tensor-shapes true   # -> no unknown-name

# Annotated is handled correctly already (proves the mechanism is forward-ref eval):
printf 'from typing import Annotated\nx: Annotated[int, "E"]\n' > /tmp/y.py
uvx pyrefly@1.0.0 check /tmp/y.py        # -> 0 errors
```

## References

- jaxtyping static-checking FAQ: <https://docs.kidger.site/jaxtyping/faq/>
- jaxtyping `_indirection.py` (the `Annotated` aliases) — installed at
  `jaxtyping/_indirection.py`; mirrored intentionally by
  [`haliax.haxtyping`](../../lib/haliax/src/haliax/haxtyping.py).
- Pyrefly tensor-shapes docs: <https://pyrefly.org/en/docs/tensor-shapes/>
- Pyrefly configuration (`tensor-shapes` key): <https://pyrefly.org/en/docs/configuration/>
- Pyrefly jaxtyping-syntax support: issues
  [#925](https://github.com/facebook/pyrefly/issues/925),
  [#743](https://github.com/facebook/pyrefly/issues/743),
  [#2332](https://github.com/facebook/pyrefly/issues/2332).
