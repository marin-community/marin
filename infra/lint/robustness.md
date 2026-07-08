# Marin lint rules — Robustness lane

Configuration explicitness and defensive error handling — where code reaches for ambient state or swallows failures instead of failing fast.

The shared detector harness — audience, detector usage, suppression markers, confidence bands, overlap precedence, output format, and self-evaluation — lives in [`shared.md`](shared.md). Read it first; it governs every lane.

## Configuration explicitness

### `ml-env-var-vs-param` — Env var used in place of an explicit parameter

**Why it's bad:** Prefer constructor/config parameters over env vars. Env
vars couple the call to ambient state, can't be type-checked, and divergent
overrides accumulate silently.

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

**Why it's bad:** Prefer explicit parameters over ambient state. Module
globals scatter configuration across the codebase and create order-of-import
bugs.

**When allowed:** True constants (frozen sets, immutable lookup tables) at
module scope are fine. The smell is *mutable* globals or globals that hold
runtime-configured state.

### `ml-magic-constant` — Magic string/number repeated without a top-level constant

**Why it's bad:** Hoist magic strings/numbers to top-level constants.
Repeated literals drift (one site updated, another not) and make searches for
"where does this value come from" yield nothing.

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

## Defensive code

### `ml-try-except-fallback` — `try/except` fallback instead of fail-fast

**Why it's bad:** Let exceptions propagate by default. Silent fallbacks
obscure whether the code is handling a real recoverable case or papering over
a bug.

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

**Why it's bad:** Never swallow exceptions unless specifically requested.
Returning `None` on parse failure makes the caller's `if result is None`
indistinguishable from "input legitimately empty."

**When allowed:** Background tasks that must keep running on per-item failure,
and best-effort teardown/cleanup where the resource is being released anyway —
but log the exception with context (and, for the loop case, emit a metric).
Silencing a failed `stop(force=True)` or `wandb.finish()` in a cleanup path is
tolerated only when the failure is of the very thing being torn down, and even
then a bare `pass` with no log is the smell — say *why* the swallow is safe.

**Bad example:**
```python
def maybe_parse(s: str) -> dict | None:
    try:
        return json.loads(s)
    except:
        return None   # raise ValueError with context, or take an explicit on_error= flag.
```

```python
def adopt_running_containers(self):
    for attempt in self._running:
        try:
            attempt.stop(force=True)
        except RuntimeError:
            pass   # cleanup-thread swallow, no log: a failed force-stop vanishes silently.
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

## Storage paths

### `ml-naive-path-join` — Ad hoc string join or slash surgery on a storage path

**Why it's bad:** Object-store keys are not normalized: `gs://b/x//y` and
`gs://b/x/y` are *different keys*, so a writer and reader that join differently
silently split the namespace (#6904, #6838). `os.path.join` on a URL,
`f"{prefix}/{path}"`, and `path.rstrip("/")`-before-join each re-solve the same
problem locally and drift. Join through `rigging.filesystem.prefix_join` (one
join) or `StoragePath` (parse once, `/` to join, `relative_to` for containment).

**When allowed:** Purely local filesystem paths that can never carry a URL
scheme (use `os.path`/`pathlib` there); appending a suffix to a known
directory-free basename; metric/logging label composition that is not a
storage key.

**Bad example:**
```python
ledger = os.path.join(cache_path, "shard_ledger.json")     # cache_path may be gs://…
record = f"{output_path}/.artifact.json"                   # doubles the slash on a trailing-/ prefix
shard = path[len(output_path.rstrip("/")) + 1 :]           # string-prefix containment; use relative_to
```

### `ml-raw-fsspec-io` — Raw fsspec handle for I/O that a `StoragePath` verb covers

**Why it's bad:** `fsspec.open`, `fsspec.core.url_to_fs`, and `fsspec.filesystem`
bypass the guarded factory in `rigging.filesystem`, so the read never charges the
cross-region transfer budget, `mirror://` is not resolved, and S3/R2 filesystems
build without the finite timeouts that stop a dead socket from wedging a shard
(#6487). Each `fs, path = url_to_fs(url); fs.<op>(path)` also re-derives the
protocol split by hand and drifts. `StoragePath` carries the guarded verbs — `exists`,
`isfile`, `isdir`, `size`, `mtime`, `ls`, `walk`, `glob`, `expand_glob`, `mkdirs`, `rm`,
`rmtree`, `rename`, `open`, `read_text`/`write_text`/`read_bytes`/`write_bytes`, and
`download_to`/`upload_from` — so a path opens, lists, and stats through one type.
(`glob` matches patterns and drops non-matches; `expand_glob` resolves a shard spec,
keeping an explicitly named literal even when it is absent.)

**When allowed:** Byte-range reads (`fs.cat_file(path, start, end)`); bulk detail
listing (`fs.ls(path, detail=True)` for a browser/report); an `fs` built with a
cache-control or backend kwarg the verbs cannot forward — `use_listings_cache=False`
for a polled read that must defeat the listing cache, `block_size`/`cache_type` that
must reach the file opener rather than the S3 constructor, or a passthrough like
`revision=`/`recursive=` on `glob`/`find`/`info`; handing a live `fs` to a library that
needs the handle (pyarrow, a streaming writer); the guarded
`rigging.filesystem.url_to_fs`/`open_url`/`filesystem` and `atomic_rename` themselves,
which are the intended low-level seam; and modules that `rigging.filesystem` itself
imports (e.g. `rigging.distributed_lock`), which sit below it in the DAG and so use raw
`fsspec` rather than depend back on `StoragePath`.

**Bad example:**
```python
fs, path = url_to_fs(output_dir); fs.makedirs(path, exist_ok=True)   # StoragePath(output_dir).mkdirs()
with fsspec.open(summary_path, "w") as f: json.dump(summary, f)      # StoragePath(summary_path).write_text(json.dumps(summary))
paths = [p for p in fs.ls(prefix) if p.endswith(".parquet")]         # ...StoragePath(prefix).ls() if str(p).endswith(".parquet")
fs.get(remote, local, recursive=True)                               # StoragePath(remote).download_to(local, recursive=True)
```

**Good example:**
```python
StoragePath(output_dir).mkdirs()
StoragePath(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True))
parquet = [p for p in StoragePath(prefix).ls() if str(p).endswith(".parquet")]
StoragePath(remote).download_to(local, recursive=True)
```
