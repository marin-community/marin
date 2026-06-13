# exp75 — contacts-v1 1.5B LR/WD/epochs tuning

Agent-driven, manually-reviewed search over `(epochs, lr, wd)` for the
contacts-v1 1.5B recipe. The launcher (`exp75_sweep.py`) trains **one explicit
point per invocation**; this doc holds the search method and the running log.
Update the log as runs finish.

## Objective

Minimize the **final-step** `eval/contacts-v1-val/loss` (unmasked LM loss on the
held-out contacts-v1 val split, read from W&B after the run completes). One
number per run. (Downstream truth is contact recapitulation; if desired, run
that eval only on the confirmed per-epoch winners — not part of the tuning loop.)

## Launching a run

One point per invocation; **keep ≤ 12 runs in flight at once** (one wave). Each
point is a separate `iris job run`:

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

## TPU & region selection

- **Region is set on the command, never in code.** Always pass
  `--region us-east5` on `iris job run` — co-located with the reused #70 caches
  and the checkpoint bucket; don't run elsewhere. The launcher sets no
  region/zone.
- **The region forces the zone.** Within us-east5 the marin preemptible pools
  have exactly one zone per family, so `--region us-east5` alone lands each slice
  in its zone — no `--zone` needed:
- **Three slices** (chosen with `-e TPU`). Global batch is 128 on all, so
  val-loss results are directly comparable across slices (grad_accum at
  `PER_CHIP_MICROBATCH=6`):

  | TPU | resolved zone | grad_accum | use |
  |---|---|---|---|
  | `v6e-8` | us-east5-b | 2 | **primary — start here** |
  | `v6e-4` | us-east5-b | 4 | fallback, **≤ 2 epochs only** (see below) |
  | `v5p-8` | us-east5-a | 2 | fallback (equal priority) |

- **Never run `v6e-4` for jobs longer than 2 epochs.** With only 4 chips and
  grad_accum 4 it is too slow for long runs. For any run > 2 epochs use `v6e-8`
  (primary) or `v5p-8`; reserve `v6e-4` for the cheap 1–2 epoch points.

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

Always ≤ 12 in flight. Review each wave before designing the next.

## Reading results

Final-step val loss per run from W&B (group `exp75-contacts-v1-tune`,
run name = trial name, e.g. `prot-exp75-cv1-1_5b-e1-lr3.5e-4-wd0p05-v1`):
the last logged `eval/contacts-v1-val/loss`. Tabulate by epoch into the grids
below.

---

## Running log

_Append wave summaries and the per-epoch confirmed optima here as runs finish._

### epochs = 1
- Grid (final-step `eval/contacts-v1-val/loss`); rows = WD, cols = LR:

  | wd \ lr | … | … | … | … |
  |---|---|---|---|---|
  | | | | | |

- Confirmed optimum: `lr=…, wd=…`, loss=…; neighbors all worse? ☐
- Notes (edges extended, zoom, etc.):

### epochs = 2
- Grid:
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
