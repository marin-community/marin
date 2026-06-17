# Zephyr import cleanup — fixing the `runpy` RuntimeWarning

**Status:** recommendation, pending review
**Tracking:** weaver #214

## Symptom

Every shard that runs under `SubprocessRunner` logs this on startup:

```
<frozen runpy>:128: RuntimeWarning: 'zephyr.runners' found in sys.modules
after import of package 'zephyr', but prior to execution of 'zephyr.runners';
this may result in unpredictable behaviour
```

## Root cause

`SubprocessRunner` runs each shard as `python -m zephyr.runners ...`
(`runners.py:286`). `runpy` executes a `-m` target in two steps:

1. import the parent package `zephyr` (run `zephyr/__init__.py`), then
2. execute the target module `zephyr.runners` under `__name__ == "__main__"`.

The problem is step 1 already imports `zephyr.runners` as a side effect:

```
zephyr/__init__.py:10   from zephyr.execution import (ZephyrContext, ...)
zephyr/execution.py:55  from zephyr.runners import InlineRunner, SubprocessRunner
```

So by the time `runpy` reaches step 2, `zephyr.runners` is already in
`sys.modules`. `runpy` warns and then **executes the module body a second
time** under `__main__`. Confirmed locally:

```
$ uv run python -c "import zephyr, sys; print('zephyr.runners' in sys.modules)"
True                                  # pulled in transitively at package import
$ uv run python -W all -c "import runpy; runpy.run_module('zephyr.runners', run_name='__main__')"
<frozen runpy>:128: RuntimeWarning: 'zephyr.runners' found in sys.modules ...
```

`runners.py` has no risky module-level side effects (only constants + class/def),
so today the double execution is **benign but wasteful** — `runpy` re-executes
`runners.py`'s own module body a second time (its class/def statements run
twice). It does *not* double the ~700 ms package cold-import cost: `zephyr` and
its module graph are imported once either way; only `runners.py`'s body is
re-run. The real issue is the structural smell it signals: `runners.py` is
simultaneously a *library module* (imported by `execution.py`) and a *script
entry point* (`python -m`). The warning is Python telling us those two roles
should not live in the same module.

## The deeper issue the warning points at

`zephyr/__init__.py` eagerly imports the **entire** module graph
(`counters`, `dataset`, `execution`→`runners`→`shuffle`/`plan`/`stage_io`,
`expr`, `plan`, `readers`, `worker_context`, `writers`) to re-export a flat
public API (`from zephyr import Dataset, ZephyrContext, col, ...`). That
eager re-export is what drags `runners` into the import that precedes the
`-m` execution. It is also the "package-level import aliasing" we generally
avoid: 93 files do `from zephyr import Dataset, ZephyrContext, counters, ...`
rather than importing from the defining submodule.

These are **two separable problems**:

- **(A) the warning** — caused specifically by a module that is both a
  library import *and* a `-m` entry point.
- **(B) the re-export aliasing** — a house-style preference, independent of
  the warning.

## Options

### Option 1 — Split the subprocess entry point out of `runners.py` (recommended)

Move the script half of `runners.py` into a dedicated leaf module that nothing
in the package-init graph imports: `zephyr/_shard_subprocess.py`. (Named for
what it does — run one shard in a child process — rather than `__main__.py`,
which would imply a user-facing `python -m zephyr` command, or
`_subprocess_runner.py`, which collides with the `SubprocessRunner` class.)

- **Move:** `SUBPROCESS_COUNTER_FLUSH_INTERVAL`, `_periodic_counter_writer`,
  `_periodic_status_logger`, `_execute_shard_subprocess`, `_subprocess_main`,
  and the `if __name__ == "__main__"` guard — i.e. everything used *only* by
  the child.
- **Keep in `runners.py`:** `_InProcessWorkerContext`, `_wrap_stage_stats`,
  `_run_stage_with_ctx` (shared with `InlineRunner`), `InlineRunner`,
  `SubprocessRunner`. `_InProcessWorkerContext` and `_run_stage_with_ctx`
  *must* stay in `runners.py` — `InlineRunner` uses them, and
  `test_shuffle.py` imports `_InProcessWorkerContext` from `zephyr.runners`.
  The new module imports those two helpers from `runners`.
- **Change the one call site:** `SubprocessRunner.execute` spawns
  `python -m zephyr._shard_subprocess` instead of `python -m zephyr.runners`.

Because `zephyr._shard_subprocess` is never imported during `zephyr/__init__.py`
(or anywhere else — only spawned as a string arg to `python -m`), `runpy`
never finds it pre-loaded → no warning, no double execution of the entry
module's body. This is the idiomatic CPython fix: separate "imported as a
library" from "run as a script." Note this does *not* reduce the package
cold-import cost — `python -m zephyr._shard_subprocess` still imports the full
`zephyr` graph once; it only removes the redundant re-run of the entry
module's body.

- **Pros:** root-cause fix; small (~1 new file, 1 changed call site, no
  consumer churn); robust against future `__init__` changes; removes the
  wasted second import in the hot subprocess path.
- **Cons:** does not address (B); `runners.py` stays in the package-init
  graph (fine — it is a normal library module again).

### Option 2 — Slim `zephyr/__init__.py` (drop the eager re-exports)

Make `__init__.py` minimal and have all consumers import from submodules
(`from zephyr.dataset import Dataset`, `from zephyr.execution import
ZephyrContext`, ...). This *also* removes the warning (the init no longer pulls
in `execution`→`runners`) and satisfies the no-aliasing preference.

- **Pros:** addresses (B); kills the warning as a side effect; no implicit
  module-graph import on `import zephyr`.
- **Cons:** **~93 files** of mechanical churn across `marin`, `levanter`,
  scripts, and tests — large review surface and merge-conflict risk for a
  benign warning. Trades a deliberate, convenient public-API surface for
  stricter import hygiene. Leaves `runners.py` as both a library module and a
  `-m` target, so the structural smell that *caused* the warning is still
  latent: re-add any transitive `runners` import to `__init__` and the warning
  returns. It treats a symptom of (A) rather than (A) itself.

### Option 3 — Both

Option 1 + Option 2. Most thorough, but the largest diff and bundles a
style refactor with a bug fix.

### Non-options (rejected)

- `warnings.filterwarnings(...)` or poking `sys.modules` — suppresses the
  signal instead of fixing the structure; explicitly against house style
  ("no ad-hoc compatibility hacks").
- Lazy `__getattr__` (PEP 562) in `__init__.py` to defer `execution` — adds
  import magic the repo otherwise avoids, and still leaves `runners.py`
  doubling as a script entry.

## Recommendation

**Do Option 1 now.** It is the precise, robust root-cause fix for the warning,
it is small and low-risk, and it makes the actual structural problem go away
(library vs. script are no longer the same module). It also drops the redundant
re-execution of the entry module's body on every `SubprocessRunner` shard
(minor — the warning, not the cost, is the real win).

**Treat Option 2 (the re-export slimming) as a separate, opt-in follow-up.**
It is a real style improvement and matches our preference against
package-level aliasing, but it is a 93-file cross-cutting change that, once
Option 1 lands, buys *no further* fix for the warning. It deserves its own PR
and its own appetite decision rather than riding along with a bug fix. If we
want it, file it as a follow-up and land it mechanically (one `from zephyr
import X` → `from zephyr.<mod> import X` codemod) with the public API surface
either dropped or kept as an explicit, lazy, documented facade.

The internal `from zephyr import counters` self-imports in `readers.py`,
`writers.py`, and `plan.py` are a minor wrinkle (they route a sibling import
back through the package `__init__`); they can be switched to
`from zephyr.counters import increment` in the Option 2 follow-up — they do
not affect the warning and are not worth a standalone change.

## Implementation sketch (Option 1)

`lib/zephyr/src/zephyr/_shard_subprocess.py` (new). The child carries **all**
imports the moved code needs — it is a standalone entry point, not a thin
shim:

```python
"""Subprocess child entry point for ``SubprocessRunner``.

Kept separate from ``zephyr.runners`` so the module run as
``python -m zephyr._shard_subprocess`` is never also imported during
``zephyr`` package initialization — otherwise ``runpy`` warns and re-executes
the entry module's body twice.
"""
from __future__ import annotations

import logging, os, sys, threading, time, traceback
from contextlib import suppress
from typing import Any

import cloudpickle
import pyarrow as pa
from rigging.log_setup import configure_logging

from zephyr.runners import _InProcessWorkerContext, _run_stage_with_ctx
from zephyr.stage_io import _ensure_picklable_exception, _stage_throughput
from zephyr.worker_context import _worker_ctx_var

SUBPROCESS_COUNTER_FLUSH_INTERVAL = 5.0
# ... _periodic_counter_writer, _periodic_status_logger,
#     _execute_shard_subprocess, _subprocess_main ...

if __name__ == "__main__":
    _subprocess_main()
```

`configure_logging(...)` stays *inside* `_execute_shard_subprocess` (not at
import time) so faulthandler installation and `os._exit` behavior are
unchanged.

`runners.py` — drop the now-unused imports (`pa`, `configure_logging`, `time`,
`threading`, `traceback`, `_ensure_picklable_exception`, `_stage_throughput`)
and change the one spawn site:

```python
proc = sp.run(
    [sys.executable, "-u", "-m", "zephyr._shard_subprocess",
     task_file, result_file, str(self._num_workers)],
    stdout=sys.stdout, stderr=sys.stderr,
)
```

Update the docstrings/comments/usage string in `runners.py` (and the
`stage_io.py` cross-reference if it names the `-m` entry) that say
`python -m zephyr.runners`.

### Verification

- `uv run python -W error::RuntimeWarning -m zephyr._shard_subprocess` — the
  **real** invocation `SubprocessRunner` uses — exits on the usage/argc error
  with **no** RuntimeWarning (today the equivalent `-m zephyr.runners` warns).
- `uv run pytest lib/zephyr/tests/test_runners.py lib/zephyr/tests/test_shuffle.py`
  — both `inline` and `subprocess` parametrizations pass (the subprocess path
  now spawns the new module; `test_shuffle` still imports
  `_InProcessWorkerContext` from `zephyr.runners`).

## Codex review

Sent the draft to `codex exec` for a critical pass. It confirmed the diagnosis
and that Option 1 is the right PR-sized fix (Option 2 is broader hygiene, not
the right vehicle for this warning). Corrections folded in above:

1. **Perf claim was overstated** — Option 1 does not cut the ~700 ms package
   cold-import cost; `python -m zephyr._shard_subprocess` still imports the
   full `zephyr` graph once. It only removes the redundant re-run of the entry
   module's body. (Fixed in Root cause / Recommendation / Option 1.)
2. **The new module needs the full child-only import set** plus the moved
   `SUBPROCESS_COUNTER_FLUSH_INTERVAL` constant — not just the two shared
   helpers. (Fixed in the sketch.)
3. **Naming** — `__main__.py` is wrong (implies a `python -m zephyr` CLI);
   landed on `_shard_subprocess` over `_subprocess_main`/`_subprocess_runner`.
4. **Verify the real `-m` command**, not only `runpy.run_module`. (Fixed.)

No pickling, logging/faulthandler, or import-cycle regressions identified, as
long as `configure_logging` stays inside the child path and the new module
stays out of the package-init graph.
