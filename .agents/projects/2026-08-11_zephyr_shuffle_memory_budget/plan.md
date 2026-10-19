# Zephyr shuffle: a unified memory budget for scatter/reduce

## Context

[PR #7941](https://github.com/marin-community/marin/pull/7941) stopped an
active OOM incident (global fuzzy-dedup, 16 GiB workers, 459 then 511
`OOMKilled` workers) by dropping the scatter flush trigger from 75% to 20% of
task memory and capping external-sort merge fan-in at a fixed 32 runs. The
author's own review comment on that PR ("worrisome, but apparently
necessary... I trust you would be able to find a more optimal value") became
[issue #7946](https://github.com/marin-community/marin/issues/7946), which
this plan closes.

The current state is four independent, mostly-guessed constants spread across
two subsystems in `lib/zephyr/src/zephyr/shuffle.py`:

| Constant | Value | Governs |
|---|---|---|
| `_SCATTER_FLUSH_THRESHOLD` | 0.20 | write-side flush trigger (fraction of task memory) |
| `_ESTIMATED_SIZE_CORRECTION_FACTOR` | 0.60 | estimated_size → RSS growth during buffering |
| `_SCATTER_READ_MEMORY_FRACTION` | 0.40 | read-side in-memory-vs-external-sort gate (untouched by #7941) |
| `_EXTERNAL_SORT_MAX_MERGE_FAN_IN` | 32 | reduce-side merge fan-in cap |

Effective write-side trigger today: `0.20 * 0.60 = 12%` of task memory —
issue #7946 documents this as a safety margin picked to unblock a production
incident, not a measured value, and notes it costs extra flushes/files on
every shuffle-heavy pipeline. Pass-1 fan-in (`ceil(sqrt(total_chunks))`) and
`POLARS_MAX_THREADS` (`ceil(task.cost.cpu)`) add two more knobs that were
never tied to memory at all.

## Decisions

- **Thread count is a fixed input, not solved jointly.** `POLARS_MAX_THREADS`
  is set as a subprocess env var before the child spawns (`runners.py`), but
  shard size is only known once the child reads scatter sidecars inside
  `run_stage` — after the thread pool already exists. Moving sidecar reads
  into the parent to make threads memory-aware too was considered and
  rejected as scope creep; threads stay CPU-derived and become a known
  constant when solving fan-in/batch-size against the remaining budget.
- **Write-side flush threshold is in scope.** It's the same class of guessed
  constant, sits directly upstream of the reduce-side fan-in decision (fewer,
  bigger flushes on the write side means fewer chunks to fan in on the read
  side), and issue #7946 is explicitly about it.

## Model

### Write side (`ScatterWriter._flush`)

Issue #7946 already specifies the reproduction recipe and reference
measurement: 1.5M rows, 150-byte payloads, 4096 target shards, production
column schema — 231.7 MiB `estimated_size()`, 507.1 MiB RSS before
`buffer.sort()`, 716.3 MiB RSS after, against a 190.9 MiB baseline. That's a
single ratio `R ≈ (716.3 - 190.9) / 231.7 ≈ 2.27`: peak RSS growth over
baseline, per byte of `estimated_size()`, measured at the worst moment
(post-sort, pre-serialize). The existing two-constant split
(`_SCATTER_FLUSH_THRESHOLD * _ESTIMATED_SIZE_CORRECTION_FACTOR`) is an
under-specified stand-in for this same ratio — phase 1 reconciles them into
one clearly-defined, measured `R`.

Once `R` is measured (and, critically, re-measured across a few shapes — row
width, target-shard count, skew — since the current 0.54–0.70 range for the
old correction factor came from only three datasets), the trigger is:

```
trigger_bytes = task_memory_bytes * safety_fraction / R
```

`safety_fraction` is an explicit, named knob (e.g. "flush is allowed to peak
at 75% of task memory") instead of being baked unrecoverably into a single
opaque multiplier — the current code has no such headroom concept at all, it
just multiplies two constants together.

### Read side (external sort merge)

Exact shard size (`shard_payload_bytes`, `total_chunks`) and `avg_item_bytes`
are already known from sidecars before the fan-in decision — no new
telemetry needed. To keep exactly one free scalar per side (see "Build vs.
benchmark" below), the read side uses the **same single-ratio shape** as the
write side rather than a multi-term formula with invented per-thread/per-row
coefficients: a streaming k-way merge holds roughly one batch resident per
input, so

```
bytes_at_risk = fan_in * batch_size_rows * avg_item_bytes
threshold     = task_memory_bytes * safety_fraction_read / R_read
fan_in        = floor(task_memory_bytes * safety_fraction_read / (R_read * batch_size_rows * avg_item_bytes))
```

clamped to `>= 2`. `batch_size_rows` (`_POLARS_STREAMING_CHUNK_SIZE`) and
`threads` (CPU-derived, decision above) are both held fixed rather than
solved — an earlier draft of this model added explicit `threads *
per_thread_overhead` and per-row-overhead terms, but those coefficients have
no measurement behind them (they'd just inherit the existing guessed `2x`/
`2x` multipliers), so folding them in would let a single fitted
`safety_fraction` silently compensate for wrong structural coefficients
instead of representing an honest margin. Pass-1 fan-in and
`max_merge_fan_in` collapse into the **same** `fan_in` value — there's no
remaining reason for two separate knobs once both are memory-derived (see
external-sort restructuring below).

`R_write ≈ 2.27` is already known from #7946 — no new measurement required to
start building. `R_read` has no prior measurement and can't be invented; it
needs one honest number (same recipe style as #7946, applied to the merge
path) before the formula means anything, but that's a single measurement, not
a sweep.

## Phases

Build first, calibrate last: everything except the two `R` coefficients and
the two `safety_fraction` values is determined by structure, not
measurement, so it's written before any benchmarking happens. `R_write`
starts from #7946's already-published `2.27`; `R_read` and both
`safety_fraction`s start as clearly-marked placeholders (conservative
defaults, e.g. reuse today's effective ~12%/~24% triggers) that phase 4
replaces with fitted values — nothing about the module's shape or call sites
changes when that happens, only the constants passed in.

### Phase 1 — Unified memory-budget module

New module (e.g. `zephyr/memory_budget.py`) with one pure function taking
`task_memory_bytes`, `task_cpu`, and shard-size stats, returning a single
result — flush threshold bytes, in-memory-vs-external-sort cutoff, fan-in,
`POLARS_MAX_THREADS`, streaming chunk size. `R_write`, `R_read`,
`safety_fraction_write`, `safety_fraction_read`, and `batch_size_rows` are its
named, overridable constants — no other free variables. `ScatterWriter.__init__`
and `ScatterReader.merge_sorted_chunks` both call this instead of each
hand-rolling its own fraction-of-`_task_memory_bytes()` arithmetic. Being a
pure function of already-available inputs, it's unit-testable without a
subprocess or cluster, and testable now with placeholder constants.

### Phase 2 — External-sort loop cleanup

Collapse `fan_in`/`max_merge_fan_in` into one parameter and make every pass
identical instead of special-casing pass 1. Once the trivial case
(`len(frames) <= fan_in`) never spills, it's the same code path a plain
`pl.merge_sorted(...).collect_batches()` call would take — so the standalone
`external_sort.py` module and the `ScatterReader.merge_sorted_chunks`
in-memory-vs-external-sort branch collapse into one private function,
`_merge_sorted_frames`, called unconditionally from `merge_sorted_chunks` and
inlined directly into `shuffle.py` (single call site, not worth a separate
module or the two single-use `write_run`/`delete_runs` closures the old
module had):

```python
def _merge_sorted_frames(frames, sort_key, external_sort_dir, fan_in, shard):
    spill_fs = spill_dir = None  # lazy: only touch external_sort_dir on an actual spill
    pass_index = 0
    while len(frames) > fan_in:
        if spill_fs is None:
            spill_fs, spill_dir = url_to_fs(external_sort_dir)
        runs = [write_run(frames[i:i+fan_in], pass_index, i // fan_in)
                for i in range(0, len(frames), fan_in)]
        frames = [pl.scan_parquet(run.url) for run in runs]
        pass_index += 1
    yield from pl.merge_sorted(frames, key=sort_key).collect_batches()
```

This also fixes a latent inefficiency: today, an input with `len(input_frames)
<= fan_in` still spills one full pass to disk before the (now-trivial) final
merge; the generic loop skips spilling entirely when nothing exceeds `fan_in`.
The lazy `spill_fs` matters here specifically because the branch that used to
gate all `external_sort_dir` filesystem access is gone — without it, every
call would pay for a filesystem round trip even when nothing ever spills.

### Phase 3 — Wiring

- `ScatterWriter`, `ScatterReader.merge_sorted_chunks`, `runners.py` (thread
  env var) all read from the phase-1 module instead of their own constants.
- Delete `_SCATTER_FLUSH_THRESHOLD`, `_ESTIMATED_SIZE_CORRECTION_FACTOR`,
  `_SCATTER_READ_MEMORY_FRACTION`, `_EXTERNAL_SORT_MAX_MERGE_FAN_IN` from
  `shuffle.py` once nothing references them.

### Phase 4 — Unit tests

Everything here runs against the phase-1 module's placeholder constants — no
cluster or calibration needed yet.

- Table-driven tests over the budget function: `(task_memory, task_cpu,
  shard_bytes, total_chunks) -> expected plan`, covering the boundaries (tiny
  task memory, huge shard, single chunk).
- Fix the existing external-sort fan-in test gap flagged in review
  ([PR #7941 comment](https://github.com/marin-community/marin/pull/7941#discussion_r3708120258)):
  the old `test_external_sort_merge_limits_later_pass_fan_in` only checked
  output correctness for one-row frames, so it would still pass if the code
  regressed to an unbounded final merge.
  `test_merge_sorted_frames_limits_every_pass_fan_in` fixes this by spying on
  `pl.merge_sorted` and asserting no call ever exceeds `fan_in`.

### Phase 5 — Calibration

The only phase that needs real hardware. Everything up to here should already
be reviewable/mergeable-behind-a-flag on placeholder constants if it's more
convenient to land in pieces.

1. Extend `lib/zephyr/tests/benchmark_shuffle.py` to accept overrides for
   `R_write`, `R_read`, both `safety_fraction`s, and `batch_size_rows`, and to
   report `ZEPHYR_WORKER_MEM_PEAK_KEY` (already emitted) alongside each
   combination.
2. Get one honest measurement of `R_read` (same recipe style as #7946,
   applied to the merge path) and reconfirm `R_write ≈ 2.27` still holds with
   current Polars/schema versions.
3. Sweep `safety_fraction_write`/`safety_fraction_read` over worker RAM,
   worker CPU, item size, target-shard count, and skew
   (`--hot-shard-frac`) — cheap cases locally, cluster-scale cases on Iris the
   way the benchmark's docstring already demonstrates (`--memory=2G` etc.).
   Include the exact shape from issue #7946 (one shard buffering ~7.2 GiB
   estimated bytes at 16 GiB task memory) as a validation point, not just a
   calibration point. Vary worker CPU independently of `bytes_at_risk`
   (`fan_in * batch_size_rows * avg_item_bytes` for reads, buffer
   `estimated_size()` for writes) specifically to check whether peak RSS
   correlates with thread count at a fixed `bytes_at_risk` — Polars' streaming
   engine is morsel-parallel (`sink_parquet`, `scan_parquet`, and
   `merge_sorted`'s streaming execution can all have multiple morsels in
   flight across the thread pool), so more threads plausibly means more
   batches resident at once, for both the write and read paths. Neither
   `R_WRITE` nor `R_READ` currently has a threads term (see "Open items"); if
   this sweep shows one is needed, add it then with a real fitted
   coefficient.
4. Replace the phase-1 placeholder constants with the fitted values. Record
   the range they're valid over; anything the sweep doesn't cover (e.g.
   extreme skew) stays covered by the safety margin, not a silent guess.

### Phase 6 — Rollout

- A `cluster`/`requires_cluster`-marked regression test (or a documented
  `benchmark_shuffle.py` invocation, per issue #7946's "Done when") that runs
  a shape close to the incident's (one shard buffering most of task memory)
  at the *calibrated* threshold and asserts no OOM kill — something #7946
  notes doesn't exist today for the write side at all. This has to wait for
  phase 5's real constants to mean anything.
- Validate against issue #7946's own bar: the fuzzy-dedup map stage still
  runs with zero OOM kills at 16 GiB task memory, and the threshold/fan-in
  numbers are traceable to a measurement rather than a margin. Close #7946
  from the landing PR.

## Open items to revisit if phase 5 contradicts the model

- If `R_write` or `R_read` turns out to depend on thread count (plausible:
  Polars' streaming engine is morsel-parallel, so more threads can mean more
  batches resident at once for `sink_parquet`/`scan_parquet`/`merge_sorted`
  alike) or on batch size, the single-ratio shape doesn't hold and the
  "threads and batch size fixed, solve fan-in" simplification breaks — the
  model needs an explicit `threads` (and/or `batch_size`) term back, with a
  real fitted coefficient, not an invented one. `R_write = 2.27` itself was
  measured without recording the reproduction's thread count, so it isn't
  validated against this risk either.
- If `R_write` or `R_read` varies more than ~2x across realistic shapes, a
  single `safety_fraction` may not be safe everywhere; may need a
  shape-dependent term (e.g. keyed off average row width) rather than one
  global constant.
