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

## Budget & quota (keep interactive roughly under cap)

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
- **Keep interactive SPENT roughly under the cap — but don't babysit it.** Over
  budget, `interactive` jobs auto-downgrade to `batch` (and a known bug can make a
  downgraded job preempt itself), so *aim* to stay under 75,000. But mild, brief
  overage is **not** a crisis — let iris bounce the overflow to batch rather than
  frantically killing runs. Only intervene (kill the least-informative cell) if
  SPENT sits well over for a sustained stretch. **The goal is to finish the sweep
  fast, not to hit a budget number.** (Resubmits are penalty-free: 10-min temp
  checkpoints + the lock-free launcher resume rather than restart.)
- **Band is a free per-launch choice (`BAND` env), independent of slice.** Any
  slice can run `interactive` (counts toward budget, not downgraded) or `batch`
  (off-budget, lower priority). Use interactive budget as you see fit — fill it
  with the best-throughput work — and run **more on batch on top** for free extra
  capacity. There is **no** "multi-host must be batch" rule.
- **Favor slices by measured throughput, re-measured.** Decide on **`wts`**
  (wall-clock tokens/sec) — see **Scheduling — finish fast** below and
  [`exp75_throughput.md`](exp75_throughput.md). The fastest slice is **not always
  the biggest**; be empirical and adaptive.

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

  | TPU | resolved zone | grad_accum | tok/s | note |
  |---|---|---|---|---|
  | `v6e-8` | us-east5-b | 2 | ~74k | fastest per run |
  | `v5p-8` | us-east5-a | 2 | ~53k | best per-chip |
  | `v6e-4` | us-east5-b | 4 | ~41k | slowest |

- **Slice choice is by measured throughput** — see **Scheduling — finish fast**
  above, and don't thrash on short waits (≥ 1 h rule below). For interactive
  single-host runs `v6e-8` is fastest per run and `v5p-8` the most chip-efficient;
  `v6e-4` is slowest (grad_accum 4), rarely worth it for long runs.

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

## Scheduling — finish fast (empirical & adaptive)

The objective is to **complete the sweep as fast as possible** while meeting the
search goals. Schedule for total throughput, not rigid rules:

- **Two bands, your discretion.** `interactive` (budget-bound, fairly scheduled, not
  downgraded) + `batch` (off-budget). Fill interactive up to ~cap with high-value work;
  pile additional work on `batch` for free. Any slice works on either band
  (`-e BAND interactive|batch`, default interactive).
- **Batch is an unfair LOTTERY — run lots, don't babysit.** On this TRC cluster batch has
  no fair scheduler: jobs are preempted constantly and routinely sit **12h+ with no
  progress — that is NORMAL, not "stuck."** Progress comes from **volume** (only users with
  many submissions win aggregate throughput), so be **aggressive**: submit freely, cap
  **~100** batch jobs (non-binding ceiling; off-quota so zero budget cost). Do **NOT**
  relocate/migrate batch cells, and do **NOT** treat no-progress as failure — only resubmit
  a batch cell when its parent **actually FAILS**. Reserve `interactive` for critical /
  near-end-of-wave cells (or ones needed fast), where fairness + budget buy reliability.
- **Favor `v5p-64` for new batch launches (user pref ~2026-06-19).** It has the top `wts`
  by a wide margin (~246k over the since-Mon window; ~323k in the clean E8 probe — ~4× v6e-8).
  Prefer it for new/relocated batch cells, *especially when few v5p-64 runs are live* (it's
  scarce, so spread a few more onto it). Next-best risk-adjusted: `v5p-32`. Still AVOID `v6e-32`.
- **Check a run's CURRENT band via iris, not budgets.** `iris --cluster marin query
  "SELECT priority_band FROM tasks WHERE job_id LIKE '<child>%'"` → **2=interactive,
  3=batch** (the **child** task; it does NOT inherit the parent driver's band, and band can
  change mid-life). Budgets only tell you total interactive spend, not a given run's band.
- **Decide on `wts`; track four metrics.** Per slice, record in
  [`exp75_throughput.md`](exp75_throughput.md): **`wts`** (wall-clock tok/s, incl.
  availability/preemption — **the decision metric**), **`ats`** (active tok/s),
  **`tts`** (training tok/s), and **MFU**. **Schedule on `wts`** unless there's a
  specific reason otherwise — it is the real progress rate, so it already folds in
  the v5p-vs-v6e speed gap *and* big-slice scarcity/preemption. The rest are
  diagnostic: `tts` is the upside a slice would give if always available, `ats`
  isolates in-run overhead, MFU is mostly irrelevant but worth a glance. Re-measure
  periodically and re-favor; never assume bigger = faster.
- **Don't oversubscribe one preemptible pool.** Spread big-slice requests across
  the v5p and v6e pools so they actually schedule (bigger = scarcer); excess just
  pends harmlessly off-budget.
- **≥ 1 h pending → relocate — INTERACTIVE/critical cells ONLY.** A *critical* child
  pending/unschedulable ≥ 1 h → move that launch to a slice that is actually scheduling.
  Do **not** apply this to batch cells — batch pending is the lottery working as intended;
  leave them queued. Never react to short waits; never migrate a healthy in-flight run.
- **Multi-host is more bug-prone.** If a multi-host run fails on a coordination bug
  (not a capacity pend / preemption), move that point to a single-host slice and
  **tell the user the bug's nature** — fix while progress continues elsewhere. (The
  `claim_and_run` lock race is already fixed/disabled — issue #6365.)

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

- Grid (**final-step** `eval/contacts-v1-val/loss`, step 4460; mid-flight evals are
  ~0.08 higher via cosine decay, so only true finals go here). Rows = WD, cols = LR.
  Filling in as runs finish (~21:15Z, 2 of 12 done):

  | wd \ lr | 8.75e-5 | 1.75e-4 | 3.5e-4 | 7e-4 | 1e-3 |
  |---|---|---|---|---|---|
  | **0.02** | 3.153 | 3.112 | 3.0623 | 3.084 | 3.087 |
  | **0.05** | 3.1535 | 3.116 | 3.0705 | **3.046** | 3.162 |
  | **0.1**  | 3.152 | 3.113 | 3.0662 | 3.052 | 3.152 |

  _**COMPLETE 15/15.** `1e-3` is worse than 7e-4 at every WD (3.087/3.162/3.152 vs
  3.084/3.046/3.052), so the LR top edge is closed at 7e-4. LR×WD interaction: optimum is
  3.5e-4 at wd0.02 but 7e-4 at wd0.05/0.1._
  _E2 update: the 1e-3 column (w2q) is now COMPLETE — 2.996/2.989/2.989 across wd0.02/0.05/0.1,
  worse than the 7e-4 column (2.980/2.979/2.970) at every WD, so the E2 LR top edge is also
  closed at 7e-4 (same as E1)._

- **Confirmed optimum: `lr=7e-4, wd=0.05`, loss=3.046; neighbors all worse? ✅** — strictly
  beats all four axis neighbors: `3.5e-4/0.05`=3.070, `1e-3/0.05`=3.162 (LR), `7e-4/0.02`=
  3.084, `7e-4/0.1`=3.052 (WD). Interior in both axes (1e-3 above, 3.5e-4 below; wd0.02 &
  wd0.1 both worse). **E1 optimum confirmed at 7e-4/wd0.05 — well above #70's 3.5e-4.**
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
- Grid (**final-step** `eval/contacts-v1-val/loss`, step 8920). Rows = WD, cols = LR.
  Filling as runs finish (~00:45Z, 1 of 15 — E2 cells are the long pole, preemption-
  stretched on batch). **Do not read mid-flight E2 values** — they're confounded by
  completion (cosine decay); only finals here.

  | wd \ lr | 1.75e-4 | 2.5e-4 | 3.5e-4 | 5e-4 | 7e-4 | 1e-3 |
  |---|---|---|---|---|---|---|
  | **0.02** | 3.047 | 3.011 | 3.001 | 2.986 | 2.980 | 2.996 |
  | **0.05** | 3.037 | 3.021 | 3.000 | 2.982 | 2.979 | 2.989 |
  | **0.1**  | 3.041 | 3.016 | 2.997 | 2.986 | 2.970 | 2.989 |
  | **0.2**  | | | | | 2.968 | |
  | **0.3**  | | | | | 2.964 | |
  | **0.4**  | | | | | **2.947** | |

  _**Base 15/15 complete** (~17:15Z). Monotone in LR (7e-4 best ~2.97–2.98 > 5e-4 > … >
  1.75e-4 ~3.04). Best E2 = `7e-4/wd0.1`=**2.970** (high LR + high WD). 7e-4 is the base top
  edge; `1e-3` column (w2q, v6e-8) still running to close it (E1's 1e-3 was worse, so 7e-4
  likely the E2 optimum too). All E2 < every E1 — 2 epochs helps. Good-LR ≥7e-4, ≫ #70's 3.5e-4._

- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐ — **NOT confirmed; WD edge
  still open.** `7e-4/wd0.2` landed ~19:30Z at **2.968**, *better* than `7e-4/wd0.1`=2.970
  (by 0.002). Then **`7e-4/wd0.4` landed at 2.947** — the WD response KEEPS DESCENDING:
  2.979→2.970→2.968→**2.947** (0.05→0.1→0.2→0.4), a clear 0.021 drop at the high end. So the
  optimum is STILL on the (now higher) edge, not interior. LR axis IS closed at 7e-4
  (5e-4/0.1=2.986, 1e-3/0.1=2.989 both worse). wd0.3 (w2u6) still finishing for the
  intermediate point. **Surprising: E2 wants high weight decay (≥0.4).** Extending further —
  **launched `7e-4/wd0.6` + `7e-4/wd0.8`** (E2, fast multi-host batch) to bracket the turn;
  accept the optimum once a higher WD lands worse than 2.947.
  _~05:43Z: budget hit 82.9k (>78k) and the over-cap downgrade was preempting the critical
  E4 7e-4 column + w2s. Shed ~16.5k by **killing the two speculative wd0.2 corners
  `5e-4/0.2` & `1e-3/0.2`** (w2t) — only `7e-4/0.2` (w2s) is needed to confirm the
  optimum; relaunch the corners cheaply only if w2s shows wd0.2 is *better* than wd0.1._
- Measured drift 1→2: lr* stable at **7e-4** (both waves); wd* nudged **up** 0.05→0.1
  (opposite the lr↓/wd↓ prior) — top of WD is flat, hence the wd0.2 check.
- Notes:

### epochs = 4
- **Center = E2 optimum region (lr 7e-4, high WD).** Tight 3×3 launched ~02:43Z (w4):
  LR {5e-4, 7e-4, 1e-3} × WD {0.05, 0.1, 0.2}, 9 cells, batch on v5p family + v6e-16
  (center 7e-4/0.1 on v5p-64). Confirms 7e-4 interior (5e-4 below, 1e-3 above) and extends
  WD to 0.2 (E2 best was on the wd0.1 edge). Expensive: 4-epoch ≈ 17.8k steps/run.
  _~07:20Z: the `1e-3/0.2` corner sat pending 4.5h on batch ("no workers match"); relocated
  to **v6e-8 interactive (w4d)** so the 3×3 completes (budget ~66k, under cap). It's the
  least-informative corner (doubly-edge), not an axis-neighbor of the center — the optimum
  check doesn't depend on it. All 9 E4 cells now in flight._
- **Controller outage ~08:46–12:10Z (~3.5h):** the iris controller query API hung (all
  RPCs DEADLINE_EXCEEDED); at ~11:00Z every active child crashed (W&B heartbeats lost) and
  when the controller returned the surviving parent drivers also FAILED on terminal child
  state (`JobFailedError: child JOB_STATE_KILLED`, failure_count=1 — outage-transient, not
  OOM/code). No progress lost (10-min checkpoints). **~12:20Z resubmitted all 9 still-needed
  cells from checkpoints on reliable single-host v6e-8** (avoiding fragile multi-host right
  after an outage): w2u=E2 7e-4/0.2 + critical E4 (7e-4/0.1, 7e-4/0.2, 5e-4/0.05, 5e-4/0.1,
  1e-3/0.1) on interactive; 3 E4 corners (5e-4/0.2, 1e-3/0.05, 1e-3/0.2) on batch. w4b
  7e-4/0.05 kept running. The 2 E2 1e-3 cells whose parents FAILED were already complete
  (recorded 2.996/2.989) — not resubmitted.
- **3 E4 corners run on BATCH; tolerate choppy progress (user guidance ~20:57Z).** This
  cluster's scheduler is weak, so **batch jobs often make little progress** — that is EXPECTED,
  not a problem. Policy for `5e-4/0.2`, `1e-3/0.05`, `1e-3/0.2` (least-informative corners —
  WD/LR edges, NOT axis-neighbors of 7e-4/0.1, so they don't gate the E4 optimum check):
  **keep them on batch; just resubmit-from-checkpoint when a parent dies — do NOT escalate to
  interactive over preemptions, do NOT relocate, do NOT panic-churn.** (Corrected ~03:30Z:
  12h+ no-progress on batch is NORMAL lottery behavior, not "stuck" — don't relocate on it;
  see the batch-lottery rule in Scheduling.) Move a cell to interactive ONLY if it's critical
  and near the end of its wave, or must finish faster. (Resubmitted on batch as w4o ~20:57Z.)
  The 7 incumbent cells (E2 gate 7e-4/0.2 +
  E4 center & 4 axis-neighbors + 5e-4/0.05) run interactive and serve all search goals.
- Grid (final-step `eval/contacts-v1-val/loss`):

  | wd \ lr | 5e-4 | 7e-4 | 1e-3 |
  |---|---|---|---|
  | **0.05** | 2.962 | 2.949 | |
  | **0.1**  | 2.958 | | 2.942 |
  | **0.2**  | | **2.938** | |

- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐ — first 3 cells in
  (~12:xxZ): 7e-4/0.05=2.949, 5e-4/0.05=2.962, **1e-3/0.1=2.942 (current best)**. Early hint
  that E4 likes HIGHER LR than E1/E2 (1e-3 best here vs 1e-3 worst at E1/E2) — watch the LR
  axis as more land; don't conclude yet (6/9 still running).
- Notes:

### epochs = 8
- **Pulled forward ~14:03Z as a speculative wave + multi-host probe (user-directed).** Wave
  `w8`. LR is pinned (7e-4 = confirmed E1 interior optimum, strong in E2), so spend cells on
  the **drifting WD axis** (E1 opt 0.05 → E2 opt 0.1 → E4 probing 0.2): grid =
  **3 LR {5e-4, 7e-4, 1e-3} × 5 WD {0.05, 0.1, 0.2, 0.3, 0.4} = 15 cells**, extending WD
  upward. Center may shift once E4 confirms; data still useful.
- **All on multi-host BATCH (off-budget), 3 cells per slice** — doubles as a multi-host
  stability/throughput re-probe (first multi-host use since the ~3.5h outage). One WD column
  per slice (promising WDs on faster slices; speculative wd0.4 on the AVOID-flagged v6e-32):

  | slice | WD col | cells (×3 LR) |
  |---|---|---|
  | v5p-64 | 0.1  | 5e-4/7e-4/1e-3 × wd0.1 |
  | v5p-32 | 0.2  | × wd0.2 |
  | v6e-16 | 0.05 | × wd0.05 |
  | v5p-16 | 0.3  | × wd0.3 |
  | v6e-32 | 0.4  | × wd0.4 |

  **Scheduling: these are BATCH — don't relocate on pending/no-progress** (that's the lottery;
  multi-host is contended and batch is off-quota/free, so just leave them queued). Only
  resubmit-on-batch when a parent **fails**. Still watch for multi-host **zombies** (iris
  "running" but no W&B/iris-log >50m) — kill+resubmit. E8 = 8×4460 = 35,680 steps/run (~2× E4).
- Grid (final-step `eval/contacts-v1-val/loss`):

  | wd \ lr | 5e-4 | 7e-4 | 1e-3 |
  |---|---|---|---|
  | **0.05** | | | |
  | **0.1**  | 2.989 | 2.950 | **2.822** |
  | **0.2**  | | | |
  | **0.3**  | | | |
  | **0.4**  | | | |

- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Notes:

### Summary across epochs
- Best confirmed `(lr*, wd*, val_loss)` per epoch; does more epochs help?
