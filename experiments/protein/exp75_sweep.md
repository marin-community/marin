# exp75 — contacts-v1 1.5B LR/WD/epochs tuning

Agent-driven, manually-reviewed search over `(epochs, lr, wd)` for the
contacts-v1 1.5B recipe. The launcher (`exp75_sweep.py`) trains **one explicit
point per invocation**; this doc holds the search method and the running log.
Update the log as runs finish.

## Status & dependencies (read first)

- **Where we are:** the launcher is **smoke-validated end-to-end on all three
  slices** (1-epoch test, `PER_CHIP_MICROBATCH=4`, no OOM, decreasing loss).
  **No real search runs have happened yet** — the Running log below is empty.
  Start with the epochs=1 wave.
- **Critical dependency — do not lose this commit.** Reusing #70's token caches
  only works because of a fix in vendored levanter
  (`lib/levanter/src/levanter/store/cache.py`, commit `a51453f1a`) that flattens
  the cache reader's exemplar with `heuristic_is_leaf`. Without it, every run
  dies at cache load with `ValueError: Sharded cache ledger missing input_ids/0
  count for shard ...`. Preserve it on any rebase / cherry-pick.
- **Throughput (measured steady `throughput/tokens_per_second` from the smoke
  runs):** v6e-8 ~74k, v5p-8 ~53k, v6e-4 ~41k → fallback priority is
  **v6e-8 > v5p-8 > v6e-4**. At ~1.05M tok/step, 1 epoch (~4,460 steps) ≈ ~18 h
  on v6e-8, ~24 h on v5p-8, ~32 h on v6e-4 — size wave wall-clock from this, and
  keep >2-epoch runs off v6e-4.
- **W&B:** entity/project come from the launch env (`eric-czech/marin`), group
  `exp75-contacts-v1-tune`, run name = trial name. Read the final-step
  `eval/contacts-v1-val/loss` there.
- **Benign log noise (ignore):** `Metadata mismatch ... preprocessor_metadata`
  (the reused caches predate that field) and a background wandb
  `log_artifact ... config.yaml FileNotFoundError` (the config-snapshot artifact
  fails to upload; metrics still log fine).
- **Microbatch:** `PER_CHIP_MICROBATCH=4` — PCM 4–6 give the identical plan; 7
  would lift v5p-8 to full 32/chip (grad_accum 1); 8 OOMs the v6e slices (a
  32 GiB chip overflows at 16/chip for this model). See the table in
  `exp75_sweep.py`.

## Objective

Minimize the **final-step** `eval/contacts-v1-val/loss` (unmasked LM loss on the
held-out contacts-v1 val split, read from W&B after the run completes). One
number per run. (Downstream truth is contact recapitulation; if desired, run
that eval only on the confirmed per-epoch winners — not part of the tuning loop.)

## Launching a run

One point per invocation. Concurrency is bounded by the **iris budget, not a run
count** — keep total SPENT under the cap with margin (see **Budget & quota**
below; the old "≤ 12 in flight" was just a rough proxy and at this chip mix runs
*over* the cap). Each point is a separate `iris job run`:

```bash
source ~/marin.env && uv run iris --cluster marin job run \
    --user "$USERNAME" --no-wait --region us-east5 --memory=1GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \
    -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
    -e TPU v6e-8 -e EPOCHS 1 -e LR 3.5e-4 -e WD 0.05 \
    -- python -m experiments.protein.exp75_sweep
```

`EPOCHS=… LR=… WD=… PREVIEW=yes uv run python -m experiments.protein.exp75_sweep`
resolves a point without submitting. Re-launching a point already trained is a
no-op (the sweep lock is keyed on the exact values), so resubmits are safe.
Monitor each job (state / logs / stop) with the
[`run-iris-job`](https://github.com/eric-czech/marin-agent-kb/blob/main/skills/run-iris-job.md)
skill; keep a wave alive (auto-resubmit failed/killed runs) with
[`monitor-sweep`](https://github.com/eric-czech/marin-agent-kb/blob/main/skills/monitor-sweep.md).

## Budget & quota (stay under the cap — check before every launch)

TPU usage is capped by a per-user **iris budget** (currently **75,000** units for
`eczech`). Cost is **dominated by chips in flight**: each task costs
`1000 × chips + ~288` (the RAM/CPU add-on is marginal), so the cap is effectively
**~75 chips total**. Per slice:

| slice | chips | cost ≈ |
|---|---|---|
| `v6e-8` | 8 | 8,288 |
| `v5p-8` | 4 | 4,288 |
| `v6e-4` | 4 | 4,288 |

- **Track actual SPENT, don't guess from a run count.**
  `uv run iris --cluster marin user budget get "$USERNAME"` (or `user budget list`)
  prints LIMIT / SPENT / MAX BAND. **SPENT is the number that matters.** The old
  "≤ 12 runs in flight" was only a crude proxy and at this chip mix actually
  *exceeds* the cap (12 runs ≈ 75.5k on their own). Read SPENT every monitor tick.
- **Other, unrelated jobs share this budget.** There are routinely non-sweep jobs
  running (e.g. 2 `dna-bolinas` jobs ≈ 8.6k as of launch). SPENT already includes
  them — so always gate on total SPENT, never on exp75 chips alone.
- **Never exceed the cap.** Over budget, `interactive` jobs (our default band) get
  downgraded to `batch` (preemptible), and a known bug then makes them
  **preempt themselves** — burning compute with no progress. Keep SPENT under
  75,000 **with margin** (target ≤ ~68k) so a new other-job or transient can't tip
  it over.
- **Before each launch:** read SPENT, add the new slice's cost, and submit only if
  the total stays under cap-with-margin. Otherwise wait for a run to finish (frees
  its chips) or pick a cheaper slice. **Resubmits are free of penalty** — the
  10-min temp checkpoints + idempotent `v1` lock resume rather than restart, so a
  run killed to stay under budget can be relaunched later with no lost data.
- **Budget-bound ⇒ chip efficiency, not wall-clock, drives slice choice.** When the
  *cap* (not the clock) is the binding constraint, a 4-chip slice fits ~2× more
  cells per budget unit than `v6e-8`, and per-chip throughput actually favors
  **v5p-8 > v6e-4 > v6e-8**. So under budget pressure prefer the 4-chip slices for
  breadth and reserve `v6e-8` for clear headroom. This *inverts* the wall-clock
  priority below — they agree only when budget is slack.
- **Batch slices are off-budget.** The larger multi-host slices run in the `batch`
  band and do **not** count toward SPENT — they add capacity *on top of* the
  interactive budget. They get an operator-set cap instead (see **Batch big-slice
  probe**). Strategy: keep interactive near the cap *and* run batch jobs on top.

## TPU & region selection

- **Region is set on the command, never in code.** Always pass
  `--region us-east5` on `iris job run` — co-located with the reused #70 caches
  and the checkpoint bucket; don't run elsewhere. The launcher sets no
  region/zone.
- **The region forces the zone.** Within us-east5 the marin preemptible pools
  have exactly one zone per family, so `--region us-east5` alone lands each slice
  in its zone — no `--zone` needed:
- **Three interactive (budget) slices** (chosen with `-e TPU`; the off-budget
  batch slices are in **Batch big-slice probe** above). Global batch is 128 on all
  slices and bands, so val-loss results are directly comparable everywhere
  (grad_accum at `PER_CHIP_MICROBATCH=4`). `tok/s` is **measured** (1-epoch smoke,
  steady `throughput/tokens_per_second` from W&B):

  | TPU | resolved zone | grad_accum | tok/s | priority |
  |---|---|---|---|---|
  | `v6e-8` | us-east5-b | 2 | ~74k | **primary — fastest, start here** |
  | `v5p-8` | us-east5-a | 2 | ~53k | 2nd choice |
  | `v6e-4` | us-east5-b | 4 | ~41k | last resort, **≤ 2 epochs only** |

- **Prioritize by throughput, gated on availability.** A run finishes sooner on
  a faster slice, so use the **fastest slice that is actually schedulable**:
  default `v6e-8`; if its pool is starved (child pending ≥ 1h, see below), drop
  to `v5p-8`, then `v6e-4`. Balance the two forces — don't sit idle waiting on
  `v6e-8` when a slower-but-free slice would finish the run sooner, but don't
  thrash on short waits either. (Re-measure if the recipe changes: pull
  `throughput/tokens_per_second` from the W&B run.)
- **Never run `v6e-4` for jobs longer than 2 epochs.** It is the slowest slice
  (4 chips, grad_accum 4); for any run > 2 epochs use `v6e-8` or `v5p-8`, and
  reserve `v6e-4` for the cheap 1–2 epoch points.

- **A started run stays put.** Once submitted on a slice, a run keeps its zone
  across preemption retries until it completes — never migrate an in-flight run.
  Slice choice applies only to **new** launches.
- **Job structure — look at the TPU *worker*, not the driver.** Each
  `iris job run` launches a lightweight **CPU driver** (the submitted id, e.g.
  `/eczech/<job-name>`) that schedules immediately and stays `running` the whole
  time; it then submits the **TPU worker** as a *child* job at
  `/eczech/<job-name>/prot-exp75-<config-id>`. The driver being `running` says
  nothing about TPU availability — it stays `running` whether the worker is
  pending, training, or about to fail. Only the **child** reflects TPU
  scheduling:
  - child `pending`/`unschedulable`, reason `Insufficient TPUs (need N, available 0)` → waiting on capacity;
  - child `running` → actually training;
  - child `failed`/`killed` → it failed (the driver then fails too).

  List both rows with `iris --cluster marin job list --prefix /eczech/<job-name>`
  and read the **child** (deeper-path) row — not the parent.
- **Flip slices on capacity starvation — but only after ≥ 1 hour.** Iris
  preemptible scheduling is slow to turn around; a **child** `pending` for a few
  minutes is normal, not starvation. Only once a new run's **worker (child)** has
  sat `pending`/`unschedulable` for **at least an hour** should you launch new
  runs on a fallback slice (`-e TPU v6e-4` or `v5p-8`, or back to `v6e-8`). Never
  react to short waits. In-flight runs are untouched. (Watch state with the
  [`run-iris-job`](https://github.com/eric-czech/marin-agent-kb/blob/main/skills/run-iris-job.md)
  skill.)

## Batch big-slice probe (extra throughput, off-budget)

Larger **multi-host** slices run in the **`batch`** band — excluded from the budget
cap (capacity *on top of* interactive) but lower priority (scheduled/preempted
after interactive) and scarcer. The slice fixes the band in code, so `-e TPU
v6e-32` is automatically batch; region is still always `--region us-east5`.

| slice | chips | hosts | pdp / grad_accum | band |
|---|---|---|---|---|
| `v5p-16` | 8 | 2 | −1 / 1 | batch |
| `v5p-32` | 16 | 4 | −1 / 1 | batch |
| `v5p-64` | 32 | 8 | −1 / 1 | batch |
| `v6e-16` | 16 | 4 | −1 / 1 | batch |
| `v6e-32` | 32 | 8 | −1 / 1 | batch |

- **Operator cap (batch is off-budget, so nothing else bounds it).** Probe started
  at 5 (one per type); **raised to 15** once multi-host proved out. Favor the
  highest realized-throughput slices — by measurement the **v5p family** (v5p-64
  best absolute ~333k tok/s, v5p-32 ~198k, v5p-16 ~102k; v6e batch MFU is only
  ~10-12% so v6e-32 ~213k / v6e-16 ~130k lag). Don't pile a whole pool onto one
  slice type — spread across the v5p and v6e preemptible pools so jobs actually
  schedule (bigger = scarcer); the ≥1h-move rule relocates anything stuck.
- **Record realized throughput** per `tpu × band` in
  [`exp75_throughput.md`](exp75_throughput.md) as runs report
  `throughput/tokens_per_second` — the reference for which slice to favor for new
  jobs, and a keepsake of band/slice efficiency at the end.
- **≥ 1 h pending → move, across bands.** Same rule as interactive: a child
  pending/unschedulable ≥ 1 h → move that *new* launch to a different slice. That
  may mean batch→interactive (if budget allows the cost), interactive→batch, or
  another batch type. Never react to short waits; never migrate an in-flight run.
- **Multi-host is more bug-prone.** If batch runs fail on a **multi-host bug** (not
  a plain capacity pend / preemption), move those cells to the single-host slices
  we know work and **notify the user with the bug's nature** — we fix it while
  progress continues on the safe slices. Do not blind-resubmit a multi-host crash
  onto the same slice type.

## Pipeline the waves (don't let them block each other)

Waves **overlap** — they are not run to completion one at a time. Each wave has
stragglers (slow slices, preemptions), and the next wave does **not** need the
prior optimum *confirmed* to start, only a good-enough leading region. As soon as
a few runs of a wave cross ~50% and give usable `eval/loss`, parameterize the next
(higher-epoch) wave speculatively from that partial signal and launch it — on the
off-budget **batch** slices when interactive budget is full — so expensive runs
are already in flight while the cheap wave finishes. **Incorporate information as
it lands:** refine/extend the next wave's grid as the prior optimum firms up, and
accept that some speculative runs are wasted (cheap insurance against serial
blocking). Confirmation (neighbor-dominance) is still required before a result is
*accepted* — but never before the next wave is *started*.

## Grid conventions (so "neighbor" is well-defined)

The search is on a discrete grid; "neighbors" are adjacent cells.

- **LR** lives on a log ladder. Coarse step ×2 per cell; refine to ×√2 (≈1.41)
  when zooming. Anchor near the #70 reference `3.5e-4`.
- **WD** lives on the ladder `{0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}` (roughly
  geometric above 0). Neighbors are adjacent entries; `0` is the bottom edge.
- A **cell** is `(lr, wd)`. Its **axis neighbors** are LR×2, LR÷2, the next/prev
  WD rung. Its **diagonal neighbors** are the four LR-step × WD-step corners.

## Per-epoch optimum confirmation (required)

An optimum at a given epoch count is **not accepted** until it is a confirmed
interior minimum of the grid:

1. **All neighbors run.** Every axis neighbor (and, before final acceptance, the
   four diagonals) of the candidate `(lr*, wd*)` must have a finished run.
2. **Strictly dominated.** The candidate's final val loss must be **strictly
   lower than every one of those neighbors**. If any neighbor ties or wins, it
   becomes the new candidate — repeat.
3. **Interior, not edge.** If the candidate sits on a grid edge (lowest/highest
   LR rung, WD = 0 or top rung), it is **not confirmed**: a better point may lie
   beyond. Extend the grid one rung in the edge direction and continue. Treat
   boundary hits as the signal to grow the box, never to stop.
4. **Optional zoom.** Once a ×2 / one-rung interior min is confirmed, optionally
   refine once (×√2 LR, intermediate WD rung) around it to sharpen `(lr*, wd*)`;
   the same neighbor-dominance test applies at the finer spacing.

Record, per epoch, the confirmed `(lr*, wd*)`, its val loss, and the neighbor
losses that establish dominance (so the evidence is auditable).

## Cross-epoch warm-start (exploit the epoch→hparam drift)

Cost scales with epochs, so spend exploration where it's cheap and extrapolate
into where it's expensive:

- **Do the wide search at epochs 1 and 2** (cheapest). Confirm interior optima
  at both per the rule above.
- **Measure the drift** `(Δlog lr*, Δlog wd*)` per epoch-doubling from those two
  rungs. Prior from AdamW+cosine: the decay timescale `∝ 1/(λ·η)` in steps, so
  doubling epochs `N` tends to **lower wd\* (~½) and nudge lr\* down**; i.e.
  `wd*·N` is roughly conserved. Use the *measured* 1→2 drift, fall back to this
  prior only if the two rungs disagree wildly.
- **At epochs 4, then 8**, seed a **tight 3×3** centered on the extrapolated
  `(lr*, wd*)` instead of a wide grid. But still **confirm** the optimum by the
  neighbor-dominance rule — expand the 3×3 if the min lands on its edge. Never
  accept an unconfirmed extrapolated guess.

This keeps each expensive (4-/8-epoch) wave to ~5–9 well-placed runs while still
producing confirmed optima.

## Wave budget sketch

| wave | epochs | shape | ~runs |
|---|---|---|---|
| 1 | 1 | coarse 4 LR × 3 WD | 12 |
| 1b | 1 | fill neighbors / extend edges to confirm | 2–6 |
| 2 | 2 | narrower grid around E1 optimum | 6–9 |
| 3 | 4 | 3×3 at extrapolated center (+ edge fills) | 5–9 |
| 4 | 8 | 3×3 at extrapolated center (+ edge fills) | 5–9 |

Concurrency gated by **budget**, not run count (see **Budget & quota**): keep
total SPENT under the cap with margin — at the current chip mix that is fewer than
12 in flight. Review each wave before designing the next.

## Reading results

Final-step val loss per run from W&B (group `exp75-contacts-v1-tune`,
run name = trial name, e.g. `prot-exp75-cv1-1_5b-e1-lr3.5e-4-wd0p05-v1`):
the last logged `eval/contacts-v1-val/loss`. Tabulate by epoch into the grids
below.

---

## Running log

_Append wave summaries and the per-epoch confirmed optima here as runs finish._

### epochs = 1
- **Wave 1 launched 2026-06-13 ~03:25Z** (`v1` lock root). Coarse 4 LR × 3 WD = 12
  runs; LR ×2 ladder anchored at #70's `3.5e-4`, WD on `{0.02, 0.05, 0.1}`.
  Slice map (interleaved, 6 v6e-8 / 3 v5p-8 / 3 v6e-4 — all 1-epoch so v6e-4 OK):

  | wd \ lr | 8.75e-5 | 1.75e-4 | 3.5e-4 | 7e-4 |
  |---|---|---|---|---|
  | **0.02** | v6e-8 | v5p-8 | v6e-8 | v6e-4 |
  | **0.05** | v6e-8 | v5p-8 | v6e-8 | v6e-4 |
  | **0.1**  | v6e-8 | v5p-8 | v6e-8 | v6e-4 |

- Grid (final-step `eval/contacts-v1-val/loss`); rows = WD, cols = LR:

  | wd \ lr | 8.75e-5 | 1.75e-4 | 3.5e-4 | 7e-4 |
  |---|---|---|---|---|
  | **0.02** | | | | |
  | **0.05** | | | | |
  | **0.1**  | | | | |

- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Notes (edges extended, zoom, etc.): _12 launched, then **2 v6e-8 cells killed
  to get under the iris budget cap** — `lr8.75e-5 × wd0.02` and `lr8.75e-5 × wd0.1`
  (lowest-LR corners, least informative). SPENT 84,044 → 67,466 (cap 75,000). The
  2 killed cells are TODO — resubmit on a cheap (4-chip) slice once budget frees,
  before E1 can be confirmed. 10 cells currently training._
  _Update ~11:48Z: an unrelated job freed budget (SPENT 67.5k→63k), so
  **resubmitted `lr8.75e-5 × wd0.02` on v5p-8** (w1b, pending on pool capacity);
  `lr8.75e-5 × wd0.1` still parked, waiting on a v6e-8 finisher for durable
  headroom._
  _Update ~13:18Z: both dna-bolinas jobs finished (all spend now exp75). Last
  parked cell **`lr8.75e-5 × wd0.1` relaunched on interactive v6e-8** → **all 12
  E1 cells now in flight**. Partial evals (half-epoch) so far: lr3.5e-4 leads
  (wd0.02 3.150, wd0.1 3.156, wd0.05 3.175) vs lr8.75e-5/wd0.05 3.216; 7e-4 &
  1.75e-4 not yet evaled._

### epochs = 2
- **Pulled forward ~13:20Z as a speculative cross** (pipelining — see that section),
  on the off-budget **batch multi-host slices**, 1 point per slice type, centered on
  the partial-E1 leading region (lr≈3.5e-4, wd flat ~0.02–0.1) with the warm-start
  prior (lr↓/wd↓ as epochs double):

  | E2 point (epochs=2) | slice | role | state @launch |
  |---|---|---|---|
  | lr 3.5e-4, wd 0.05 | v6e-32 | center | pending (gang: need 8 hosts) |
  | lr 2.5e-4, wd 0.05 | v5p-64 | LR− | pending (no workers free) |
  | lr 5e-4,   wd 0.05 | v6e-16 | LR+ | pending (no workers free) |
  | lr 3.5e-4, wd 0.02 | v5p-32 | WD− | **running** (multi-host OK) |
  | lr 3.5e-4, wd 0.1  | v5p-16 | WD+ | pending (no workers free) |

  First multi-host run (v5p-32) initialized `jax.distributed` cleanly — multi-host
  path works. Availability is the expected bottleneck (bigger = scarcer). The cross
  gives all 4 axis-neighbors of the center → start of the E2 interior-optimum check;
  extend as E1 firms up and the higher-LR (7e-4) E1 evals land.
- **Update ~15:03Z — cap raised 5→15, 10 more E2 batch jobs (`w2d`)** to fill out a
  full **5 LR × 3 WD** grid: LR ∈ {1.75e-4, 2.5e-4, 3.5e-4, 5e-4, 7e-4} × WD ∈
  {0.02, 0.05, 0.1}. High-LR points (5e-4, 7e-4 — most decision-relevant given E1's
  optimum is ≥3.5e-4) placed on the fastest slices (v5p-64, v6e-32); v5p favored 7/10
  (best throughput + MFU), 3 on the v6e pool to avoid oversubscribing one pool. All
  10 pending at launch (scarcity); ≥1h-move relocates any stuck.
- Grid (final-step `eval/contacts-v1-val/loss`):
- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Measured drift 1→2: `Δlog lr=…, Δlog wd=…`
- Notes:

### epochs = 4
- Extrapolated center: `lr≈…, wd≈…`
- Grid:
- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Notes:

### epochs = 8
- Extrapolated center: `lr≈…, wd≈…`
- Grid:
- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Notes:

### Summary across epochs
- Best confirmed `(lr*, wd*, val_loss)` per epoch; does more epochs help?
