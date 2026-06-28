# Debug Midtrain — val-loss crossover

## Orientation (read first)

**What this doc is:** the task logbook for debugging the Delphi midtraining val-loss *crossover* /
"1e22 miss" — small-ladder scaling-law extrapolations miss 1e22 on the iso-FLOP ladder (raw-val actual
lands 12–19% *below* the predicted loss, an acceleration), and re-scoring on the Jaccard-0.5
decontaminated val reportedly flips 1e22 to *worse* (the crossover). It holds the experiment plan
(#1–#7), the executed work, and a handoff. To pick up: skim this Orientation, then jump to
**Experiment status (#1–#7)** and **HANDOFF FOR NEXT AGENT** near the bottom.

**Status (2026-06-11):** Exp #7 (iso-token ladder) is done for **budget #1 (1B tokens)** → result:
**no crossover** on the raw val (strictly monotonic in scale, 1e22 best at 0.9176). The 9 checkpoints
exist (W&B project `delphi-midtraining`, runs `delphi-{base}-p33m67-tok1b-lr50`). Quantified the same
day with the published report's Chinchilla endpoint fit (see the **2026-06-11 overlay entry**): fit
through 3e20, the prediction error at 1e22 is **−3.8%** on iso-token versus **+18.6%** on the matching
iso-FLOP recipe (signed: pred vs actual) — the "1e22 miss" does not survive token-matching. Also that day a long-standing mode
claim was **CORRECTED**: the iso-FLOP 1e21/1e22 K=0.20 runs are CPT from ~fully-decayed checkpoints,
NOT mid-WSD cooldown resumes (see the CORRECTION entry; the visualization logbook still carries the
wrong claim). The most direct line to the *original* "1e22 worse at Jaccard 0.5" question is
**exp #1** (decontaminated re-eval) — not yet run.

**Companion docs (this investigation depends on them):**
- [`nemotron_math_data.md`](nemotron_math_data.md) — the **DATA reference**: the byte-identical val
  contract (why cross-scale losses are comparable), the dedup pipeline, and the **contamination
  analysis** + artifacts (`scratch/verified_pairs/`, `scratch/contaminated_val_ids.json`, the per-scale
  Jaccard-threshold tables). Exp #1 and #4 build directly on it.
- [`delphi_midtraining_visualization.md`](delphi_midtraining_visualization.md) — the
  **scaling/prediction analysis**: where the iso-FLOP "1e22 miss"/acceleration (the crossover this doc
  debugs) is quantified, plus the interactive report and its builder scripts. **Caveat:** its claim
  that the iso-FLOP 1e21/1e22 runs "resumed mid-cooldown" is **wrong** — see this doc's 2026-06-11
  CORRECTION entry before relying on its mode-confound arguments.
- `.agents/projects/delphi_midtraining.md` — project-level plan/state.
- `scripts/analysis/delphi_*` — the analysis + report builders, incl.
  `delphi_isotoken_endpoint_scaling.py` (iso-token vs iso-FLOP endpoint-fit overlay, 2026-06-11; writes
  `sk_midtrain_analysis_fable/`, caches W&B exports in `midtrain_wandb_data/`, both gitignored).

## 2026-06-08 — crossover investigation

### Problem

Why is there a cross over? Why is 1e22 doing worse at 0.5? There should just be
a shift up, and there shouldn't be cross-over.

For duplicates.

### Hypothesis

- We have a bug somewhere
- There is model specific weirdness for 1e22 model that causes higher loss
- We decontaminated too much and the val loss is now noisy

### Experiments

- Plot jaccard threshold and see how val loss changes / where inflection point happens (understanding / bug)
- Visualize logprobs across delphi model (model specific weirdness)
- Have a proper validation set from 4plus on nemotron and see how we do there (noise)
- Decontamination val set against 3 to try and decouple difficulty from decontamination.
- Look at train loss i.i.d. from 1e22 and then train loss from the first X number of documents.
- Run 1e23 for p33m67 lr 0.33x at remainder of nemotron math outside val set and see if we're still off-trend.
- Pick three budgets, let's train the ladder models on these fixed token budgets and then see if that exhibits.

---

## 2026-06-08 — #7 implementation: iso-token ladder

**What #7 tests.** In the current ladder each model is trained **iso-FLOP**
(K=0.20 of its *own* pretrain tokens), so the math-token budget D grows with
model size — 0.245B at 3e18 up to 32B at 1e22 (130x spread). That confounds
model size N with tokens D. Holding **D fixed** (iso-token) isolates N: if 1e22
still crosses its neighbours at the same D, the crossover is an N effect; if it
disappears, it was a token-budget artifact. Running the whole ladder as **CPT**
was also intended to remove a second confound the visualization logbook flags —
it claims the small ladder is CPT (fresh warmup) while 1e21/1e22 "resumed
mid-WSD". **CORRECTED 2026-06-11: that claim is false** (see the CORRECTION
entry below). The iso-FLOP 1e21/1e22 K=0.20 runs are *also* CPT, MODEL_ONLY
from ~fully-decayed end-of-pretrain checkpoints, so the iso-FLOP ladder was
already mode-consistent; the only residual differences the iso-token ladder
removes are init source (final GCS checkpoint vs HF export) and warmup length
(~3% vs 10% of steps).

### Vehicle: CPT fixed-token, whole ladder, one mode

`BudgetPolicy.fixed_tokens(N)` already exists; `resolve_cpt_budget` turns it
into `num_train_steps = round(N / (batch_size * 4096))` per base. CPT streams
public HF weights (`initialize_from_hf`), pinned for *every* scale 3e18->1e23,
so the whole ladder runs in one mode. The byte-identical math val set (12,500
windows) is preserved automatically via `data_section_override`, so val loss
stays comparable across scales.

Verified the architecture heuristic (`_model_config_for_base`, which derives the
config from `hidden_dim` only) reproduces the registered configs exactly:
3e20->23 layers (control), 1e21->26, 1e22->37. So CPT-from-HF loads cleanly for
the big bases.

### What was wired

`experiments/midtrain_specs/delphi_small_cpt_k020.py` (+ `default_budget_label`
re-export in `lib/marin/src/marin/midtraining/__init__.py`):

- New `--budget-tokens N` flag -> `BudgetPolicy.fixed_tokens(N)`. Default stays
  `pretrain_fraction(0.20)`; mutually exclusive with `--probe-steps`.
- Extended `BASES` / `DEFAULT_TPU` / `ALLOWED_TPUS_PER_BASE` with **1e21**
  (default v5p-64; allows up to v5p-512) and **1e22** (default v5p-256; allows
  v5p-128/256/512) — the canonical slices those runs used. CPT has the same
  MODEL_ONLY/fresh-optimizer HBM footprint as those prior K=0.20 runs
  (corrected 2026-06-11: they were CPT too, not full-state resumes), so these
  slices are safe.
- Iso-token cells carry `tok<label>` in the id (e.g.
  `delphi-1e22-p33m67-tok1b-lr50`) and tags `sweep:delphi-cpt-isotoken` +
  `budget:tok1b`, so they never collide with or pollute the K=0.20 sweep.
- Default K=0.20 path (id `...-k0p20-...`, tag `sweep:delphi-small-cpt-k020`) is
  unchanged; existing CPT/budget/identity tests pass.

### Step counts @ 1B math tokens (mix p33m67, lr 0.5) — real, from dry-run

| base | params | batch | steps @ 1B | K=0.20 steps | K=0.20 tokens |
|---|---|---:|---:|---:|---:|
| 3e18 | 447M | 8 | 30,518 | 7,467 | 0.245B |
| 9e18 | 550M | 16 | 15,259 | 8,863 | 0.58B |
| 2e19 | 837M | 16 | 15,259 | 11,025 | 0.72B |
| 3e19 | 998M | 32 | 7,629 | 7,603 | 1.0B |
| 9e19 | 1.4B | 64 | 3,815 | 8,057 | 2.1B |
| 2e20 | 1.9B | 64 | 3,815 | 11,295 | 3.0B |
| 3e20 | 2.5B | 128 | 1,907 | 7,102 | 3.7B |
| 1e21 | 3.4B | 512 | 477 | 4,411 | 9.25B |
| 1e22 | 9.7B | 1024 | **238** | 7,647 | 32.1B |

The K=0.20 token column matches the per-scale exposure table in
`nemotron_math_data.md`. Key consequence: at a fixed 1B the small models get a
*huge* dose (3e18 ~ 83% of a full pretrain) while 1e22 barely moves (238 steps,
~0.6% of pretrain). So 1B is "generous to small, stingy to big" vs iso-FLOP — a
cheap small-D anchor + pipeline smoke. The big model only gets meaningful
adaptation at larger D, which is why the experiment is **three budgets**:
sweep the token axis and see where (if) the crossover survives.

### Plan

1. **Budgets:** 1B / 4B / 16B (geometric x4). 1e22's iso-FLOP point is 32B, so
   add **32B** as a tie-point where iso-token meets iso-FLOP for 1e22.
2. **Cells:** 9 bases x {1B, 4B, 16B} = 27 runs, mix p33m67, lr 0.5 (+9 for 32B).
3. **1e23:** deferred — it has HF weights but no `DelphiModel` yet (needs
   batch/LR/steps registered; same prerequisite as #6).

### How to run (one cell per invocation; loop in the driver shell)

Dry-run first:

```bash
.venv/bin/python experiments/midtrain_specs/delphi_small_cpt_k020.py \
  --base 1e22 --mix p33m67 --lr 0.5 --budget-tokens 1000000000 --dry-run
```

Sweep one budget (loop bases in the shell, never in the script):

```bash
for b in 3e18 9e18 2e19 3e19 9e19 2e20 3e20 1e21 1e22; do
  .venv/bin/python experiments/midtrain_specs/delphi_small_cpt_k020.py \
    --base $b --mix p33m67 --lr 0.5 --budget-tokens 1000000000
done
```

### Caveats / pre-launch

- Readout = `math_val_loss` on the byte-identical 12,500-window val set (W&B
  project `delphi-midtraining`); overlay iso-token curves on the iso-FLOP ladder.
- Watch host RAM on the 1e22 HF export (default `--ram 256g`); bump if exit-137.
- Status: **wired + dry-run verified; NOT launched** — awaiting greenlight.
- Pre-existing (not this change): the cooldown test
  `test_true_cooldown_rendered_data_section_bit_identical_to_reference` is red on
  this branch — `build_cooldown_spec` missing `zone` / `per_device_parallelism` /
  `temp_save_interval`. Confirmed unrelated by stash-revert.

### Launch log

**2026-06-09 02:55Z (19:55 PT 06-08) — first iso-token cell launched (1e22 @ 1B, v5p-32).**

- DRI: Ahmed. Budget #1 of three (1B); smoke + small-D anchor.
- Coordinator (CPU) iris job: `/ahmedah/delphi-isotoken-1e22-tok1b-20260608-195543`
  (`--cpu 1 --memory 3GB --disk 9GB --no-preemptible --region us-east5,us-central1`,
  `-e WANDB_API_KEY <scrubbed>`), running the launcher which submits the v5p-32 child.
- Command: `python experiments/midtrain_specs/delphi_small_cpt_k020.py --base 1e22
  --mix p33m67 --lr 0.5 --budget-tokens 1000000000 --tpu v5p-32`.
- Cell / W&B id: `delphi-1e22-p33m67-tok1b-lr50-a001` (project `delphi-midtraining`).
- Output root (pinned): `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-tok1b-lr50-a001`.
- init_from: `marin-community/delphi-1e22-9.7Bparams-160Btokens@ca7b0e7c0a6b` (CPT, model-only).
- Final step: 238 (1B tokens / (1024×4096)). LR triangular, 10% warmup.
- Lineage: HEAD `0e9f281a` + dirty patch `0d08f86a` (iso-token launcher), shipped via
  the working-tree bundle (`git ls-files --cached --others`, content read from disk; 21.1 MB).
- **Val set: held out (verified 4 ways).** Rendered `data:` block bit-identical to the
  1e21 reference; `num_validation_sequences=12500` + `shuffle_before_trainval_split=true`
  (feistel PRNGKey(0)); Levanter `train_sets()` = slice `[0, N−12500)`, val = `[N−12500, N)`
  → disjoint; prior empirical reconstruction in `nemotron_math_data.md`.
- **Caveat — HBM:** v5p-32 is below 1e22's prior allowlist floor (v5p-128) and the
  logbook has "1e22 HBM failed" notes. Launched per operator (capacity = v5p-32);
  watching for an early HBM OOM after the HF load / model+optimizer alloc.
- Babysit: tight early checks (OOM/capacity window ~5–10 min), then 15-min cadence.

**2026-06-09 ~03:16Z — pending ~20 min on v5p-32 capacity (not a config error).**
- Child: 4/4 tasks pending; coordinator healthy (running).
- Pending reason: `Coscheduling: need 4 workers ... only 0 of 4 have capacity:
  Insufficient memory (need 256.0GB, available 32.8GB)` + `Autoscaler: Waiting for
  workers in scale group 'tpu_v5p-preemptible_32-us-east5-a' to become ready`.
- Diagnosis: the "32.8GB" is the currently-ready **CPU** workers, not v5p hosts. The
  v5p scale group uses `ct5p-hightpu-4t` = **448 GiB RAM** workers (`marin.yaml:101`),
  so the 256 GiB request fits. Autoscaler status: that group has **`ready: 0`** and is
  provisioning **preemptible** v5p-32 in us-east5-a — pure capacity wait, GCP hasn't
  handed over a slice. Do NOT lower `--ram` (request is correct).
- Open question for DRI: wait on preemptible autoscale, or target reserved v5p-32 /
  a different zone? Both reserved and preemptible v5p groups show `ready: 0` right now.
- DRI: keep waiting on preemptible.

**2026-06-09 ~03:23Z — allocated; all 4 tasks running on preemptible v5p-32.**
- Slice came up ~5 min after the wait decision; child `running`, 4/4 tasks.
- Logs confirm the p33m67 mix cache loads (nemotron_cc {medium,medium_high,medium_low},
  math `4plus-2c5519`, proofpile_2, starcoderdata) and **`Splitting dataset into
  train/val sets. Shuffle before split: True`** — live confirmation of the val carve-out.
- Benign `Metadata mismatch` warnings (cache `preprocessor_metadata` append_bos/max_length;
  same canonical caches all delphi runs use). No OOM, no errors.
- Next: HF weight load (~19 GB) + JAX compile of 9.7B on v5p-32 → step 1 (the HBM moment).

**2026-06-09 ~03:37Z — 1e22 v5p-32 FAILED: int32 activation-indexing overflow (not HBM-full).**
- `JaxRuntimeError: RET_CHECK ... allocation_size_words <= int32_max; 2327838720; f32[37,64,4096,3840]`
  (layers × per-device-batch × seq × hidden). Per-device batch 64 (1024/16 chips), sharded /16 →
  2.33 B words > 2.147 B int32 limit.
- Root cause: the new CPT launcher didn't microbatch; the prior 1e21/1e22 K=0.20 launcher
  (`exp_delphi_math_10b_midtrain.py`, per_device_parallelism=4 on its 1e22 runs) did.
- **Fix (this change):** added `--per-device-parallelism` to the CPT launcher
  (`ComputeProfile.per_device_parallelism` → `levanter_config.py:101`). Trajectory-neutral
  (global batch stays 1024; grad accumulation). pdev16 → 0.58 B words, fits.

**2026-06-09 ~03:53Z — pivoted to full 1B ladder on free us-east5; big bases on v5p-8 + grad accum.**
- User: "find whatever compute is free, queue up, stay us-east5"; then "lots of v5p-8 — run even 1e22
  there with cranked grad accum (it's a 9B model)."
- v5p-8 = 1 `ct5p-hightpu-4t` host (4 chips, ~380 GB HBM, 448 GB RAM). 9.7B model+Adam ≈ 136 GB fits
  ~3:1; small microbatch dodges int32 + shrinks activations. Tradeoff: compute-bound, ~12–20 h for
  1e22 1B (vs ~3–5 h v5p-32) — grad accum makes it FIT, not FAST.
- Launched 9-base ladder, 1B / p33m67 / lr0.5, us-east5: 3e18–3e19 → v5p-8; 9e19/2e20/3e20 → v5p-16;
  1e21 → v5p-8 pdev16 (a002); 1e22 → v5p-8 pdev8 (a003). Killed broken 1e22 a001 + the v5p-32 a001/a002.
- Status: 3e18/9e18/2e19 children RUNNING on v5p-8; 3e19 coord running; 9e19/2e20/3e20/1e21/1e22
  coordinators pending on us-east5 CPU autoscale.
- Open: does 1e22 fit v5p-8 at pdev8? Verdict pending (coordinators still spinning up). Monitoring.

**2026-06-09 ~04:35Z — GOAL set: drive all 9 to finish, us-east5-a, interactive priority. Bottleneck = us-east5 CPU coordinator cap.**
- 4 small bases (3e18/9e18/2e19/3e19, `--cpu 1` coords) TRAINING on v5p-8 in us-east5-a; 3e18 logging
  `loss 3.12→1.61→2.17`. Children confirmed on `marin-tpu-v5p-preemptible-8-us-east5-a-…` (preemptible).
- 5 others (9e19/2e20/3e20/1e21/1e22) couldn't get a CPU coordinator. Resubmitted at `--cpu 0.5
  --priority interactive --zone us-east5-a` on v5p-8 (mid bases moved off v5p-16); STILL pending —
  pool `cpu_vm_e2_highmem_2_ondemand-us-east5-a` is `at_max_slices`, **0 cores free**. Hard cap.
- Consequence: plentiful free v5p-8 TPU sits idle while CPU coordinators queue. Each coordinator
  blocks for the full run (the launcher must outlive its child), so us-east5 CPU can host only ~4-5
  concurrent cells. Tension: "stay us-east5-a" vs using the free v5p-8 (needs coordinators, CPU-capped).
- Asked DRI: serialize in us-east5-a (compliant, slow) vs coordinators in us-central1 + training in
  us-east5-a (parallel, uses the free v5p-8). [child priority/preemptibility still TODO]
- DRI chose: stay 100% us-east5-a (serialize). Babysitting to completion.

**2026-06-09 ~23:40Z — 5/9 done; serialized tail in progress.**
- Succeeded: 3e18, 9e18, 2e19, 3e19, 9e19. Running: 2e20 (`loss ~1.5`, **preemptions=4**, self-healing
  on preemptible v5p-8). Queued: 3e20, 1e21, 1e22.
- us-east5-a CPU pool scaled DOWN to 4 workers after the small bases finished; now `0 cores free,
  at_max_slices` again → strict 1-coordinator-at-a-time. Each queued base waits for the prior to free CPU.
- Watch: 1e22 (last, ~12-20h once it starts) may face many preemptions on preemptible v5p-8 → restart
  churn (HF reload + recompile per preempt). If it thrashes, consider non-preemptible (scarcer reserved cap).
- TODO at seal: verify each succeeded run's final math_val_loss in W&B (don't trust orchestrator success alone).

**2026-06-10 ~06:13Z — VERDICT: 1e22 FITS v5p-8 at pdev8. The int32 microbatch fix is validated.**
- 1e22 stepping: `step 4/238, loss 1.55`, checkpointing (temp step-2). No int32/OOM. So a 9.7B
  CPT-from-HF model trains on ONE 4-chip v5p-8 host with `per_device_parallelism=8` (grad accum,
  global batch 1024) — the fix that died at pdev=64 on v5p-32 now works.
- Throughput: **~451.8 s/step → ~29 h remaining (~30 h total)** for the 238-step run. 9.7B on 4 chips
  is compute-bound; grad accum makes it fit, not fast (as flagged).
- Concurrency recovered: 3e20/1e21/1e22 running together (CPU loosened). Ladder seals when 1e22 does
  (~2026-06-11 ~12Z), plus any preemption overhead (1e22 is preemptible; each preempt = HF reload +
  recompile + resume from temp ckpt — recoverable but adds time).
- Speedup if wanted: v5p-32 + pdev (now validated) ≈ ~7.5 h (4× fewer chips→steps). Offered to DRI.

**2026-06-10 ~07:35Z — 1e22 (a003) FAILED: host-RAM OOM (exit 137), NOT HBM.**
- `Exit code 137: OOM killed (container exceeded memory limit)`, failures=4, during
  `jax.array_serialization` checkpoint commit. On v5p-8 (single host), serializing the full 9.7B
  model+Adam (~136 GB) to GCS exceeded the 256 GB container limit. pdev8 fit was fine (stepped to ~14).
- Note: step-2 serialized OK at 256g but a later commit OOM'd → likely host-RAM growth over steps
  (data-loader buffering / fragmentation), not just a fixed peak.
- Fix: relaunched 1e22 as **a004 on v5p-8 with `--ram 400g`** (worker has 448 GB).
  `delphi-iso-1e22-tok1b-v5p8b-20260610-003904`. Watching whether it holds through serializations.
- Escalation: if it OOMs again at 400g, v5p-8 single-host can't serialize 9.7B reliably → move to
  **v5p-32** (4 hosts → ~34 GB/host serialization: robust AND 4× faster). Re-recommended to DRI.

**2026-06-10 ~10:13Z — 1e22 a004 (`--ram 400g`) FIX CONFIRMED: no OOM, checkpointing cleanly.**
- a004 at step ~23/238, serialized checkpoints at step-2/5/8/11/14/17/20 (every ~3 steps) — **past
  the ~step-14 point where a003 OOM'd at 256g, with NO OOM.** Host-RAM serialization blocker resolved;
  v5p-8 is now reliable for 1e22 too. So the canonical CPT `--ram` default (256g) is just too low for 9.7B.
- ~451 s/step → ~27 h remaining (~30 h total). v5p-32 remains a 4× speed option, no longer a reliability need.
- 3e20 (~10 h elapsed) and 1e21 (~4.5 h) still running on v5p-8, no OOM (≤3.4B fit 256g fine).

**2026-06-10 ~21:40Z — 8/9 sealed; 1e22 a004 HUNG/crashed at step 99 → moved to v5p-32.**
- Sealed: 3e18/9e18/2e19/3e19/9e19/2e20/3e20/1e21 (all ≤3.4B, clean on v5p-8).
- 1e22 a004 stalled at step ~99 for ~3.5 h: iris said `running` (zombie) but **W&B said `crashed`,
  last update 220 min ago, _step=99, loss=1.43**. A preemption (preemptions 1→2) mid-eval killed the
  process and the single-host recovery hung. Worker logs stopped at 18:24Z (mid-eval iter 133/893).
- **This is the 3rd distinct v5p-8 failure for 1e22 (slow → host-RAM OOM → preemption-hang)** — all
  from cramming 9.7B onto one fragile preemptible host. Took the DRI's standing v5p-32 offer.
- Relaunched 1e22 fresh on **v5p-32**, a005, pdev16, `--ram 400g`, us-east5-a, interactive:
  `delphi-iso-1e22-tok1b-v5p32-20260610-145521`. ~7.5 h (4× faster), serialization sharded over 4 hosts.
- Lesson: single-host v5p-8 is fine up to ~3.4B but the wrong tool for a 9.7B preemptible run; the
  9.7B model wants the multi-host slice (distributed serialization + shorter wall-clock = less preempt exposure).
- Now watching W&B freshness too (iris `running` alone missed the zombie).

**2026-06-10 ~22:35Z — v5p-32 move FAILED (us-east5-a tier_blocked) → reverted 1e22 to v5p-8 (a006).**
- 1e22 v5p-32 a005 child pending ~36 min: `No workers match constraints [...zone...]` +
  `Autoscaler: tier_blocked`. The only v5p-32 preemptible autoscale group is in **us-central1-a, not
  us-east5-a** → v5p-32 unobtainable in us-east5-a now (quota tier cap).
- Reverted to v5p-8: a006, `--ram 400g`, pdev8, us-east5-a → `delphi-iso-1e22-tok1b-v5p8c-20260610-153637`.
  The proven in-region slice. ~30 h; preemption-hang recoverable via relaunch.
- Net: 1e22 (9.7B) as a 1B CPT smoke is genuinely awkward on available us-east5-a capacity — v5p-8 is
  slow+preemption-fragile, v5p-32 is quota-blocked in-region. The 8 smaller bases sealed clean on v5p-8.
- Detour cost: a006 restarts fresh (a004's ~99 steps redone). Faster paths need v5p-32 quota in-region
  or us-central1 (cross-region data) — DRI's call.
- Monitoring a006 with step-advance hang detection (step unchanged ~30 min while "running" ⇒ hang ⇒ relaunch).

**2026-06-11 ~00:55Z — 1e22 BLOCKED: us-east5-a TPU quota `tier_blocked` (both v5p-8 + v5p-32).**
- a006 (v5p-8) pending ~2.3 h: `Autoscaler: tier_blocked: quota-pool tier monotonicity`. v5p-32 us-east5-a
  same. 0 v5p jobs running (capacity reclaimed). CPU fine (e2-highmem-2 ready:5). So **no us-east5-a TPU is
  allocatable for 1e22 right now** — a cluster quota state, not transient capacity, beyond my remit
  (no cluster/quota changes per ops rules).
- 8/9 bases DONE (the bulk of the iso-token ladder). 1e22 (9.7B) is the lone holdout.
- Escalated to DRI. a006 held queued (auto-allocates if the tier_block clears). Options: wait /
  us-central1 (cross-region data, needs explicit OK) / admin quota fix.

**2026-06-11 ~01:16Z — DRI: run 1e22 in us-central1 using HF. Added cross-region opt-in.**
- DRI's key correction: this is CPT, so the model loads from **HF** (`initialize_from_hf`), not GCS —
  so the big cross-region cost I worried about (9.7B model from GCS × preempt-reloads) doesn't exist.
  Only the small tokenized data is read cross-region.
- Framework enforced region co-location in TWO guards: `spec._assert_run_region_alignment` and
  `preflight` component-region check. Both blocked us-central1 compute + us-east5 data caches.
- Change: added `--region` flag to the launcher + a narrow `allow_cross_region_data` opt-in to
  `validate_midtrain_spec` and `preflight`, active **only when `--region` is overridden**. Default
  behavior (in-region) is unchanged. So: model from HF (free), data read from us-east5 cross-region
  (~tens of GB, DRI-OK'd), compute + checkpoints **local** to us-central1 (no big cross-region writes).
- Launched: `delphi-iso-1e22-tok1b-c1-20260610-181644`, a007, **v5p-128, us-central1**, pdev16, ram400g,
  interactive. ~2 h (4× v5p-32). Output `gs://marin-us-central1/.../a007`. Val byte-identical (data
  block unchanged us-east5). Watching allocation + first step.

**2026-06-11 ~03:10Z — v5p-preemptible broadly unavailable; 1e22 queued in us-central1 (a008).**
- a007 (v5p-128 us-central1) `tier_blocked` (too-big tier). Dropped to **a008 (v5p-32 us-central1)**:
  reason `Autoscaler: Waiting for workers to become ready` — NOT tier_blocked, but ~1 h pending with no
  preemptible v5p-32 capacity materializing.
- So preemptible v5p is broadly constrained right now: us-east5 tier_blocked (v5p-8/32), us-central1
  v5p-128 tier_blocked, us-central1 v5p-32 no-capacity. GCP preemptible scarcity + a quota tier state —
  beyond my control (no cluster/quota changes per ops rules). Stopped churning slices.
- a008 held queued (us-central1 v5p-32) — auto-runs the moment preemptible capacity returns. **8/9 DONE.**
  Escalated. Options: wait / reserved (non-preemptible) v5p / admin look at the v5p-preemptible tier_block.
- `delphi-iso-1e22-tok1b-c1b-20260610-190723` / a008. Region override + cross-region opt-in all working;
  pure capacity wait now.

**2026-06-11 ~05:52Z — a008 ALLOCATED + STEPPING on us-central1 v5p-32. Move validated.**
- Preemptible v5p-32 us-central1 capacity returned ~05:45Z (after ~2.5 h wait); a008 stepping (`step 0`).
- End-to-end us-central1 path validated: `--region` override (compute+output us-central1), cross-region
  HF model + us-east5 data read, pdev16 (int32 fit), ram400g (distributed serialization over 4 hosts).
- ~7.5 h to finish (238 steps @ ~115 s/step). Then **all 9 iso-token bases sealed.** Riding to completion.

**2026-06-11 ~13:09Z — SEAL: all 9 iso-token bases DONE. Result: NO crossover.**
- 1e22 a008 finished step 238 on us-central1 v5p-32 at 13:09Z. All 9 W&B runs `finished`.
- Final `math_val_loss` (1B tokens, p33m67, lr 0.5), byte-identical val set across all scales:
  3e18 1.2648 · 9e18 1.2163 · 2e19 1.1596 · 3e19 1.1382 · 9e19 1.0899 · 2e20 1.0523 ·
  3e20 1.0300 · 1e21 0.9993 · **1e22 0.9176**.
- **Strictly monotonic in scale; no crossover; 1e22 best (1e21→1e22 is the LARGEST drop, −0.082).**
  This is the opposite of the iso-FLOP "1e22 miss" → strong evidence the original crossover/miss was a
  token-budget confound of iso-FLOP (D grows with N), not a real 1e22 issue. At fixed D, N-scaling is clean.
- W&B runs: `delphi-{3e18..1e21}-p33m67-tok1b-lr50-a001/a002` (us-east5 v5p-8/16),
  `delphi-1e22-p33m67-tok1b-lr50-a008` (us-central1 v5p-32).
- Caveats: (1) this is budget #1 of 3 (1B); 4B + 16B still pending for the full #7. (2) These are RAW
  (contaminated) val losses — the original "1e22 worse at Jaccard 0.5" needs the decontaminated re-eval
  (exp #1) on these checkpoints. (3) 1e22 ran cross-region (us-central1 compute, us-east5 data, HF model).
- Launcher additions this session (lint+type clean, uncommitted): `--budget-tokens` (iso-token),
  `--per-device-parallelism`, `--region` + `allow_cross_region_data` opt-in (spec.py + preflight.py),
  1e21/1e22 bases + v5p-8/16/32 allowlists.

**2026-06-11 ~13:20Z — queued in-region re-run (a010, us-east5a) for placement compliance.**
- Original /goal said "stay us-east5a"; 1e22 actually ran us-central1 (a008) after the user authorized
  that when us-east5a v5p went quota tier_blocked. Stop hook flags the literal placement gap.
- Queued **a010 in us-east5a** (`delphi-iso-1e22-tok1b-east5-20260611-061826`, v5p-32, pdev16, ram400g,
  in-region — no cross-region) to ALSO train 1e22 in us-east5a and satisfy the constraint. Result is
  region-independent (byte-identical val) → reproduces 0.9176; us-central1 a008 stays the deliverable.
- Status: a010 `No workers match` — us-east5a v5p still tier_blocked. Queued; babysitting; auto-runs when
  the block clears (capacity outside my control). DRI can `cancel a010` (us-central1 already authorized).

---

## 2026-06-11 — endpoint-fit overlay (iso-token vs iso-FLOP) + mode CORRECTION

### What was built
- `scripts/analysis/delphi_isotoken_endpoint_scaling.py` (lint-clean, uncommitted): applies the
  published report's exact endpoint fits (imports `fit_floor_power`/`fit_log_linear` from
  `build_delphi_midtraining_interactive_report.py`) — Chinchilla floor+power
  `L(C) = E + A*(C/1e18)^(-alpha)` and the 2-param log-log — to the iso-token sweep, with the
  iso-FLOP K=0.20 ladder (matching mix/LR recipe) overlaid. Fit on 3e18→3e20, 1e21/1e22 held out;
  fit-through / fit-type / budget controls mirror the published page; output HTML is fully
  self-contained (Plotly inlined, no CDN — sandboxed previews need that).
- Data pulled fresh from W&B `marin-community/delphi-midtraining` into a local export cache
  `midtrain_wandb_data/` (~2.5 GB, gitignored; same per-run schema as `download_midtrain_wandb.py`;
  finished runs never refetched). Iso-token = tag `sweep:delphi-cpt-isotoken`; iso-FLOP =
  `delphi-{scale}-{mix}-k0p20-lr{lr}-a*` (3e18→3e20) + `delphi-1e2{1,2}-{mix}-{9p25b,32p07b}-lr0.*-*`.
- Outputs in `sk_midtrain_analysis_fable/` (gitignored): interactive HTML, overlay PNG,
  `isotoken_endpoints.csv`, `isoflop_k020_endpoints.csv`, `isotoken_scaling_fits.csv`.
- Validation: the K=0.20 p33m67/lr50 fit reproduces the published payload's parameters exactly
  (E=0.1595, A=1.4426, alpha=0.1138) and its held-out errors — same data, same code path.

### Result (fit through 3e20, p33m67 / lr 0.5)
| held-out | iso-FLOP K=0.20: actual → pred (err) | iso-token 1B: actual → pred (err) |
|---|---|---|
| 1e21 | 0.7935 → 0.8167 (**+2.9%**) | 0.9993 → 0.9775 (**−2.2%**) |
| 1e22 | 0.5610 → 0.6652 (**+18.6%**) | 0.9176 → 0.8829 (**−3.8%**) |

Log-log slope −0.045 (iso-token) vs −0.098 (iso-FLOP): about half the apparent iso-FLOP "scaling"
is the growing token budget. At fixed D the 1e22 point is on-trend (slightly *above* the
extrapolation — mild flattening, as a floor would predict); on iso-FLOP it lands 18.6% *below* it
(the "miss"/acceleration). Sign flip + ~5x error collapse ⇒ the miss is the D-grows-with-N
confound, not a 1e22 effect. Rebuild: `cd scripts/analysis && python
delphi_isotoken_endpoint_scaling.py [--use-cache]` (needs `WANDB_API_KEY` unless `--use-cache`).

### CORRECTION — iso-FLOP 1e21/1e22 K=0.20 runs are CPT, NOT "cooldown resumed mid-WSD"
Both this logbook (the #7 rationale, as originally written) and
`delphi_midtraining_visualization.md` ("small ladder = false-midtrain fresh warmup; 1e21/1e22 =
true-midtrain resumed mid-cooldown") claimed the iso-FLOP 1e21/1e22 points trained in a different
mode. **That claim is wrong.** Verified from the W&B run configs (cached in `midtrain_wandb_data/`):

- `delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d`: `initialize_from_checkpoint_path =
  mirror://adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-38206`,
  `checkpoint_init_mode = MODEL_ONLY` (fresh optimizer), own `warmup=250` + linear `decay=7397` to 0.
  The base run (W&B `marin-community/marin`, **finished**) is 38,235 steps, linear schedule, decay
  over the last 20% to `min_lr_ratio=0` → at step-38206 the LR is ~0.4% of peak: **fully cooled.**
- `delphi-1e21-p33m67-9p25b-lr0.5-efbc63`: same pattern; init `step-21979` of 22,057 → ~98.2% decayed.
- The k0p20 small ladder initializes from the released HF models — the same cooled endpoints.

So the whole published iso-FLOP K=0.20 ladder is **mode-consistent CPT from the cooled end of
pretraining**; the 1e21/1e22 cells differ only in init source (final GCS checkpoint vs HF export)
and warmup length (~3% vs 10%). The genuinely mid-WSD family is the separate
`delphi-true-*-cooldown20-*` batch (see `midtraining_delphi.md` 2026-05-24) — likely the source of
the conflation. Consequences:

1. The iso-FLOP vs iso-token comparison (exp #7) is mode-clean: the "1e22 miss" cannot be attributed
   to a CPT-vs-cooldown step-0 inconsistency, strengthening the token-budget-confound conclusion.
2. The viz logbook's argument that Kaiyue's improvement-decomposition fails partly *because* of the
   step-0 mode inconsistency needs revisiting (the baseline inconsistency it cites is far smaller
   than claimed). **The viz logbook has NOT been corrected** — fix it or read it with this in mind.
3. The wrong claims in THIS file (the #7 rationale, the BASES/HBM bullet, the "cooldown launcher"
   reference, and the miss direction in the HANDOFF Goal) were corrected in place on 2026-06-11,
   each marked "corrected 2026-06-11".

---

# HANDOFF FOR NEXT AGENT — iso-token ladder (debug-midtrain exp #7)

## Goal
Original problem (top of this file): scaling-law extrapolations from the small ladder **miss 1e22** on
the Delphi **iso-FLOP** ladder — on raw val, 1e22's actual loss lands 12–19% *below* the predicted loss
(an acceleration; "the 1e22 miss"), and the Jaccard-0.5 decontaminated re-scoring reportedly flips 1e22
to *worse* (the crossover; exp #1's question). Hypothesis: it's a **token-budget confound**
(iso-FLOP gives each scale a different D, so N and D are tangled). **Exp #7** tests this: train the whole
ladder at a **fixed token budget** (iso-token) and see if the miss/crossover persists. This session ran
**budget #1 = 1B tokens**, mix p33m67, lr-factor 0.5, across all 9 bases 3e18→1e22.

## Result — NO crossover
Final `math_val_loss` (1B tokens, byte-identical val set across all scales):
| 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---|---|---|---|---|---|---|---|
|1.2648|1.2163|1.1596|1.1382|1.0899|1.0523|1.0300|0.9993|**0.9176**|
**Strictly monotonic in scale; 1e22 best; 1e21→1e22 is the LARGEST drop (−0.082)** — the opposite of the
iso-FLOP miss. ⇒ the original crossover was a token-budget/iso-FLOP confound, not a real 1e22 problem.
W&B: project `delphi-midtraining`, runs `delphi-{base}-p33m67-tok1b-lr50-a00{1,2,8}`.
Endpoint-fit quantification (2026-06-11 overlay entry): same Chinchilla fit-through-3e20 as the published
report → iso-token held-out errors **−2.2% (1e21) / −3.8% (1e22)** vs iso-FLOP (p33m67/lr50)
**+2.9% / +18.6%**; log-log slope −0.045 (iso-token) vs −0.098 (iso-FLOP). Both ladders are
mode-consistent CPT (see CORRECTION entry), so the contrast is clean. Artifacts:
`scripts/analysis/delphi_isotoken_endpoint_scaling.py` → `sk_midtrain_analysis_fable/`.

## Exact launch commands
All runs go through a CPU coordinator (`iris job run`) that runs the one-cell launcher. Setup once:
```bash
cd <worktree>; ln -sf ~/code/marin/.marin.yaml .marin.yaml   # WANDB key for workers
export WANDB_API_KEY="$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('.marin.yaml'))['env']['WANDB_API_KEY'])")"
URL=http://localhost:10000   # IAP tunnel to iris-controller-marin must be up on :10000
```
Coordinator wrapper (one per base; `--job-name` must be FRESH each submit):
```bash
.venv/bin/iris --controller-url=$URL --cluster=marin job run \
  --cpu 0.5 --memory 3GB --disk 9GB --no-preemptible --priority interactive --zone us-east5-a --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" --job-name delphi-iso-<base>-tok1b-<TS> \
  -- python experiments/midtrain_specs/delphi_small_cpt_k020.py <LAUNCHER-ARGS-BELOW>
```
The 9 SEALED cells (launcher args; coordinator wrapper above):
| base | launcher args | slice/region |
|---|---|---|
| 3e18 | `--base 3e18 --mix p33m67 --lr 0.5 --budget-tokens 1000000000 --tpu v5p-8` | v5p-8 us-east5 |
| 9e18 | `--base 9e18 … --tpu v5p-8` | v5p-8 us-east5 |
| 2e19 | `--base 2e19 … --tpu v5p-8` | v5p-8 us-east5 |
| 3e19 | `--base 3e19 … --tpu v5p-8` | v5p-8 us-east5 |
| 9e19 | `--base 9e19 … --tpu v5p-8` | v5p-8 us-east5 |
| 2e20 | `--base 2e20 … --tpu v5p-8` | v5p-8 us-east5 |
| 3e20 | `--base 3e20 … --tpu v5p-8 --per-device-parallelism 16` | v5p-8 us-east5 |
| 1e21 | `--base 1e21 … --tpu v5p-8 --per-device-parallelism 16 --attempt 2` | v5p-8 us-east5 |
| 1e22 | `--base 1e22 … --tpu v5p-32 --per-device-parallelism 16 --ram 400g --region us-central1 --attempt 8` | **v5p-32 us-central1** |
(1e22 coordinator used `--region us-central1` + `-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1`; everything else default region us-east5. `…` = `--mix p33m67 --lr 0.5 --budget-tokens 1000000000`.)

## 1e22 saga (the gotchas — read before re-running a 9.7B cell)
1e22 (9.7B, batch 1024) is the only awkward cell. In order, what broke and the fix:
- **int32 activation overflow** on v5p-32 at default per-device batch (64): `f32[37,64,4096,3840]` sharded
  /16 > 2^31 words ⇒ `--per-device-parallelism 16` (microbatch). [a001 died here]
- **host-RAM OOM** (exit 137) during checkpoint serialization on a single v5p-8 host (`--ram 256g` too
  low for 9.7B+Adam ≈136 GB) ⇒ `--ram 400g`. [a003 died, a004 fixed it then…]
- **preemption-recovery hang/crash** on single-host preemptible v5p-8 (slow ~30 h + zombie at step 99,
  W&B `crashed` while iris said `running`). [a004]
- **us-east5 v5p `tier_blocked`** (quota-pool tier monotonicity, all sizes) ⇒ DRI authorized us-central1.
- **v5p-128 us-central1** also `tier_blocked` (too-big tier); **v5p-32 us-central1** autoscaled fine.
- ✅ **a008: 1e22 on v5p-32 us-central1, pdev16, ram400g → SUCCEEDED** (math_val_loss 0.9176).
- us-east5a placement re-runs (a010 preemptible → preempt-crash; a011 reserved `--no-child-preempt` →
  coordinator RPC fail) were attempted only to satisfy the literal "stay us-east5a" goal; **abandoned**
  (killed) — placement-only, identical result. **us-east5a is hostile to this 9.7B run right now.**
  **DRI explicitly accepted the us-central1 result and directed killing the in-region re-runs** ("if
  science is done kill that"). 1e22's us-east5a placement is a **deliberately-waived constraint**, not
  an open item — the data is region-independent (byte-identical val), so re-running buys nothing.

## Launcher/code changes this session (lint+type clean, UNCOMMITTED)
`experiments/midtrain_specs/delphi_small_cpt_k020.py` + `lib/marin/src/marin/midtraining/{spec,preflight,__init__}.py`:
- `--budget-tokens N` (iso-token budget; cell id carries `tok<label>`); added 1e21/1e22 bases + v5p-8/16/32 allowlists.
- `--per-device-parallelism N` (microbatch → int32 fit on small slices).
- `--region R` + `allow_cross_region_data` opt-in in `validate_midtrain_spec`/`preflight` (active only when
  `--region` is overridden): CPT loads model from HF, data block stays us-east5 (byte-identical val), only
  compute+checkpoints move region. Lets a quota-blocked region be bypassed.
- `--no-child-preempt` (reserved/non-preemptible TPU child).

Also uncommitted (2026-06-11 analysis session): `scripts/analysis/delphi_isotoken_endpoint_scaling.py`
(endpoint-fit overlay; see the 2026-06-11 entry) and `.gitignore` entries for `midtrain_wandb_data/` +
`sk_midtrain_analysis_fable/`.

## State
- **All 9 iso-token bases sealed.** 1e22 is in us-central1 (`gs://marin-us-central1/checkpoints/…-a008`);
  the other 8 are in us-east5. No jobs running.
- Code is uncommitted — **commit + PR** the launcher/spec changes if keeping them.
- Ops gotchas: us-east5 CPU coordinator pool caps at ~4–5 long-lived coordinators (use `--cpu 0.5`);
  v5p quota `tier_blocked` flips under heavy launch/kill churn; preemptible v5p single-host is fragile for 9.7B.

## Experiment status (#1–#7) — what's done, what's next
The 7 experiments from the original plan (top of this doc). Only **#7** has run; its 9 iso-token
checkpoints are now the substrate several of the others need. "(contam)" = uses the contamination
analysis + artifacts in [`nemotron_math_data.md`](nemotron_math_data.md).

| # | tests | status | how the next agent runs it |
|---|---|---|---|
| **7** | iso-token: does the crossover persist at fixed D? | **DONE @ 1B → no crossover** (monotonic; endpoint fit: 1e22 −3.8% vs iso-FLOP +18.6%) | Run budgets **4B + 16B**: identical to the 9 cells (see HANDOFF commands), swap `--budget-tokens 4000000000` / `16000000000`. Confirms monotonicity holds as D grows. Rebuild the overlay after: `scripts/analysis/delphi_isotoken_endpoint_scaling.py` (picks new budgets up by tag). |
| **1** | raw vs **decontaminated** val loss across Jaccard thresholds — the direct line to "1e22 worse at 0.5" (contam) | **NOT RUN — highest value** | Re-score the 9 checkpoints (+ the iso-FLOP runs) on the val *decontaminated at a sweep of Jaccard cutoffs*. Efficient: dump **per-window loss once** per checkpoint over the 12,500 windows, then re-average over the non-contaminated subset at each threshold **offline**. Inputs: `scratch/verified_pairs/` (per-pair Jaccard), `scratch/contaminated_val_ids.json`, the per-threshold tables in `nemotron_math_data.md`. |
| **4** | decontaminate val against ONE reference (budget-3 = 3e20 exposure) to hold difficulty constant (contam) | NOT RUN | Use the **per-scale exposure** map `scratch/nemotron_math_isoflop_contamination_exposure.{json,csv}`; build one fixed decontaminated val from the 3e20 reference, score all budgets on it. |
| **2** | model-specific weirdness: per-window logprobs for 1e22 | NOT RUN | Dump per-token/per-window loss for the 1e22 checkpoint (vs a smaller budget); split by contaminated vs clean windows (contam). |
| **3** | noise: a freshly fuzzy-deduped val from 4plus | NOT RUN | Carve a clean val from 4plus *after* a fresh fuzzy-dedup (pipeline in `nemotron_math_data.md`), re-eval — the clean-val control. |
| **5** | train-side: 1e22 train loss i.i.d. vs first-X docs | NOT RUN | Decontamination-independent diagnostic; compare 1e22 loss on an i.i.d. sample vs its first X training docs. |
| **6** | new point: train 1e23 (p33m67, lr 0.33×) on held-out math outside val | NOT RUN | Needs a `DelphiModel` for 1e23 first (HF weights exist via `DELPHI_1E23_HF_REPO`; register batch/LR/steps). Then a hero-scale run on the held-out math remainder. |

---

## 2026-06-12 — /goal: iso-token budgets tok500m + tok2b (500M first, then 2B)

**~06:05Z — tok500m ladder LAUNCHED: all 9 cells on RESERVED v4 in us-central2-b.**
- Goal: both ladders to completion, sealed per budget, strict order (500M before any 2B cell).
- Capacity scan found 102 idle RESERVED v4-8 slices + 2 idle v4-32 in us-central2-b (v5p-8 us-east5-a
  degraded health=0.05; us-east5-a CPU pool at max). Reserved = no preemption risk. DRI's standing
  directive: use any free compute (model loads from HF, no locality constraint).
- Launcher edits for this: `default_budget_label` now yields `500m` (was `0p50b` — keeps run names
  `tok500m` and the analysis-script regex happy); ALLOWED_TPUS_PER_BASE extended with v4-8/16/32/64 +
  v6e-16/32 (1e22) entries. Dry-runs verified all 9 cell ids + step counts.
- Cells (all `--mix p33m67 --lr 0.5 --budget-tokens 500000000 --region us-central2 --no-child-preempt`,
  coordinators `--zone us-central2-b --cpu 0.5 --priority interactive`, ram default 256g):
  3e18/9e18/2e19/3e19 → v4-8 (auto pdev); 9e19/2e20/3e20 → v4-8 pdev8; 1e21 → v4-8 pdev4;
  1e22 → v4-32 pdev4 (16 chips × 32GB = 512GB; int32 per-shard ≈0.15e9 words, safe).
- Coordinators: `/ahmedah/delphi-iso-{base}-tok500m-20260612-0604xx/0605xx`.
- Expected final W&B _step: 15258/7628/7628/3814/1906/1906/953/237/118 (3e18→1e22).
- Watch points: first v4 run of this launcher (new type), us-central2-b CPU pool autoscale from 0,
  reserved-capacity match via --no-child-preempt (flag's first real use; a011's earlier failure was
  coordinator RPC, unrelated).

**~06:45Z — ALL 9 tok500m cells STEPPING on reserved v4 (us-central2-b). v4 path validated.**
- Allocation took ~10 min total (reserved pool, no quota friction — vs 2.5 days of v5p saga at 1B).
- W&B: 3e18 s1199, 9e18 s759, 2e19 s379, 3e19 s189, 9e19 s46, 2e20 s46, 3e20 s24, 1e21 s4, 1e22 s0;
  losses 1.7-2.4, consistent with the 1B ladder's early trajectory. No int32/OOM at the chosen pdev
  (v4-8 auto/pdev8/pdev4; 1e22 v4-32 pdev4). Reserved capacity => zero preemption exposure.
- Monitoring at 30-min cadence to completion.

**~07:40Z — sweep 2: all healthy; re-sliced 3e20 + 2e20 to v4-16 (accum-bound on v4-8).**
- Marginal rates (steps/min): 3e18 38, 9e18 24, 2e19 19, 3e19 7.1, 9e19 3.5, 1e21 0.45, 1e22 0.3 →
  ETAs 4-8h. Outliers: 3e20 0.6/min (ETA ~24h, accum 4 on 4 chips) and 2e20 2.4/min (~12.5h).
- Strict order gates tok2b on the SLOWEST cell → killed a001 of 3e20 (s49) + 2e20 (s140), relaunched
  attempt 2 on idle reserved v4-16 pdev8 (accum 2/1): ETA ~6h/~2h. Coordinators
  `delphi-iso-{3e20,2e20}-tok500m-20260612-0706xx`. Lesson for tok2b sizing: 4× steps → use v4-16/32
  for mid bases and bigger for small-batch cells.

**~08:15Z — sweep 3: evals explain mid-ladder slowdowns; a002 cells still compiling.**
- 9e19 (s189) / 1e21 (s24) / 1e22 (s15) each slowed to d~1-3/sweep EXACTLY at 10% progress = the
  mid-training eval pass (12,500-window val, slow on 4-16 chip slices). Heartbeats fresh — NOT zombies.
- 2e20/3e20 a002 (v4-16): children running ~25 min, W&B not stepping yet (HF load + compile). Verify next sweep.
- 3e18 marginal rate dropped 38→11/min (eval pass in window?); if sustained, ETA ~17h and 3e18 becomes
  the ladder gate (batch-8 step-bound run; v4 ~2x slower/chip than v5p). Hold; reassess next sweep.

**~08:50Z — sweep 4: all 9 stepping; a002 cells live on v4-16; eval-dip confirmed transient.**
- 2e20 a002 s93, 3e20 a002 s49 (v4-16 first steps ~60min post-submit — cold-slice init, no relaunch needed).
- 3e18 back to 40/min (s4599; the sweep-3 dip was its eval pass). 9e19 resumed (d=93). 1e21/1e22 slow
  but moving (d=8/9; eval-heavy cells). ETAs ~3.5-15h; no zombies; no action.

**~09:25Z — sweep 5: all 9 healthy; ladder gate ~9-11h (9e19/2e20/1e22).**
- Rates (steps/min): 3e18 40, 9e18 22, 2e19 16, 3e19 8.1, 9e19 2.7, 2e20 2.7 (v4-16), 3e20 1.3 (v4-16,
  mid-eval), 1e21 0.46, 1e22 0.086 (mid-eval at s24=20% boundary). Eval-at-10%-boundary dips are the
  recurring benign pattern. Intermediate math_val_loss already logging (1e22 1.130 at s27).

**~10:00Z — sweep 6: 2e20/1e22 recovered post-eval; 3e20+1e21 on re-slice watch.**
- d/sweep: 3e18 1400, 9e18 950, 2e19 570, 3e19 285, 9e19 94, 2e20 139 (v4-16 ~4/min ETA ~6.6h),
  3e20 35 (1/min — eval-polluted?), 1e21 2 (mid-20%-eval; eval-dominated on v4-8), 1e22 11 (ETA ~4.3h).
- Rule for next sweep: 3e20 <2/min or 1e21 <0.35/min effective => re-slice to idle reserved v4-32.

**~10:40Z — sweep 7: re-sliced 3e20 to v4-32 (a003); rest on track.**
- 3e20 sustained 1.3/min on v4-16 in a clean window (<60/sweep rule) => killed a002 (s175), relaunched
  a003 on v4-32 pdev4 (`delphi-iso-3e20-tok500m-20260612-090956`). 1e21 cleared its bar (d=17, ETA ~6h).
- ETAs: 3e18/9e18 ~2.7h, 2e19 4h, 3e19 4.4h, 9e19 ~7h, 2e20 ~6h (eval window), 1e21 ~6h, 1e22 ~3.7h,
  3e20 ~3h post-reslice. Ladder gate ≈ 9e19/2e20 → seal ~17:00-17:30Z.

**~11:15Z — sweep 8: 1e22 zombie flag = FALSE ALARM (38.9GB HF export); 3e20 a003 stepping on v4-32.**
- 1e22 s49 frozen + hb 34m tripped the zombie rule, but worker logs show it mid eval+checkpoint+HF-export
  cycle at step-50 (38.9GB upload @79MB/s, 8min; tasks 4/4 alive, 0 failures/preemptions). Its per-10%
  cycle = eval + ckpt + HF export ≈ 30-45min silent. RULE ADJUSTED: 1e21/1e22 zombie threshold 60min.
- 3e20 a003 (v4-32 pdev4): s74 within ~30min of allocation — re-slice paid off (~2.5/min incl. init).
- d/sweep: 3e18 1600 (s10399, 68%), 9e18 897 (83%), 2e19 680, 3e19 290, 9e19 122, 2e20 141. ETAs:
  9e18 ~1.5h, 3e18 ~3h, 2e19 ~5h, 3e19 ~5h, 9e19 ~5.9h, 2e20 ~5.6h, 3e20 ~3h, 1e21 ~5h, 1e22 ~5h.

**~11:45Z — sweep 9: all healthy; 9e18 at 95%; first finishes expected next sweep.**
- 1e22 resumed post-export (s60). 3e20 a003 d=57 incl. its 10%-eval — clean-window rate TBD; no more
  re-slices for it (restart cost > remaining savings). Gates: 3e20/2e20/9e19 ≈ 7-8h → seal ~19-20Z.

**~12:20Z — sweep 10: FIRST CELL SEALED — 9e18 finished (_step 7628, math_val_loss 1.28925).**
- Sanity: tok500m 1.2892 > tok1b 1.2163 (less adaptation at half budget) ✓.
- 3e18 88% (~1.2h). 3e20 clean-window 1.9/min on v4-32 → ~6.6h = ladder gate. 2e20 ~5.5h, 9e19 ~6.3h.

**~15:55Z — sweep 17: 4/9 done; 1e22 preempted at s117/118 (!), self-healing.**
- DONE: 3e18 1.34541, 9e18 1.28925, 2e19 1.23164, 3e19 1.20874 (exact steps, monotonic).
- 1e22 W&B "crashed" at s117 = iris preemption by /larry/* job (preemptions=1; priority preemption hits
  reserved slices too). Tasks auto-restarted 39min ago: HF reload + recompile + temp-ckpt resume (~s116)
  + final step/eval/export ≈ 30-60min. Per protocol: no intervention, preemptions self-heal.
- Tail: 9e19 s1644 (~1.4h), 2e20 s1597 (~1.7h), 3e20 s649 (~2.4h), 1e21 s177 (eval window, ~2-3h).

**~16:25Z — sweep 18: 1e22 FINISHED post-preemption-resume (0.97412). 5/9 sealed.**
- Preemption self-heal validated end-to-end: temp-ckpt resume -> final step -> final eval/export.
- DONE so far: 3e18 1.34541, 9e18 1.28925, 2e19 1.23164, 3e19 1.20874, 1e22 0.97412.
- Remaining: 9e19 91%, 2e20 90%, 1e21 82%, 3e20 74% — all healthy, seal ~17:30-18Z.

**15:57 UTC (real) — status update + timestamp correction.**
- NB: sweep-entry timestamps above from "~08:15Z" onward drifted ahead of real UTC by up to ~1h (they
  were estimated by cadence, not clock; sweeps actually ran more often due to goal-hook turns). Job-name
  timestamps (e.g. 20260612-0604xx = launch ~06:05 UTC) are accurate. Entries from here use real UTC.
- tok500m: **5/9 finished+verified** — 3e18 1.34541, 9e18 1.28925, 2e19 1.23164, 3e19 1.20874,
  1e22 0.97412 (all exact expected steps; monotonic in scale; all above their tok1b counterparts as
  expected at half budget). In flight: 2e20 s1905/1906 + 9e19 s1879/1906 (both in final/late eval-export,
  ~minutes-30min), 3e20 s849/953 (~1h), 1e21 s207/237 (~1-1.5h incl. final eval).
- Checkpointing verified in GCS (gs://marin-us-central2/checkpoints/delphi-*-tok500m-*): temp ckpts
  every ~3 steps + full HF exports at 10% milestones + final (1e22: hf/step-{50,100,118}).
- Events this session: 2 deliberate re-slices (2e20 v4-8->v4-16 a002; 3e20 v4-8->v4-16 a002->v4-32 a003),
  1 preemption (1e22 s117, auto-recovered), 1 false zombie (1e22 38.9GB HF export window), 0 quota stalls.
- Next: last 4 finish -> tok500m SEAL (logbook + delphi_isotoken_endpoint_scaling.py rerun + fit-error
  report) -> tok2b launch (strict order): 3e18/9e18/2e19 v4-16 auto-pdev; 3e19/9e19 v4-16 pdev8;
  2e20/3e20/1e21/1e22 v4-32 pdev4. tok2b ETA ~12-24h after launch.

---

## 2026-06-14 — SEAL: tok500m (9/9) DONE + tok2b (1e21,1e22) DONE

**Canonical W&B metric:** `eval/nemotron_cc_math_v1/4plus/loss` (the byte-identical 12,500-window math
val; what the logbook calls "math_val_loss"). The W&B `summary["math_val_loss"]` key is empty — read
`eval/nemotron_cc_math_v1/4plus/loss` from the summary instead.

### tok500m SEALED — all 9 bases finished (final 4plus/loss, exact steps)
| 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---|---|---|---|---|---|---|---|
|1.34541|1.28925|1.23164|1.20874|1.15586|1.11538|1.09256|1.06156|**0.97412**|
**Strictly monotonic in scale; no crossover; 1e22 best; 1e21->1e22 the largest drop (-0.087).** Same clean
result as tok1b — confirms the iso-token monotonicity is not a single-budget artifact. Per-base ordering
tok500m > tok1b > tok2b holds for every scale (more math tokens -> lower val), e.g. 1e22: 0.97412 (500M) >
0.9176 (1B) > 0.85640 (2B); 1e21: 1.06156 > 0.9993 > 0.93687. Winning attempts:
2e20=a002, 3e20=a003, all others a001 (the aNNN < winner were preemption-killed on reserved v4 us-central2-b).
All on RESERVED v4 us-central2-b (v4-8/16/32, the re-sliced 2e20/3e20 on v4-16/32); coordinators interactive.

### tok2b — 1e21 + 1e22 DONE (the user-directed subset, NOT the full ladder)
| run | budget | final 4plus/loss | slice | steps |
|---|---|---|---|---|
| delphi-1e21-p33m67-tok2b-lr50-a001 | 2B | **0.93687** | v5p-32 us-east5-a (pdev16) | 954 |
| delphi-1e22-p33m67-tok2b-lr50-a003 | 2B | **0.85640** | v5p-32 us-east5-a (pdev16, ram400g) | 477 |
DRI directed launching ONLY 1e21+1e22 at 2B (the two longest), interactive, "largest free compute any
region" (CPT loads from HF, region-free). The other 7 tok2b cells (3e18..3e20) were NOT run. Both 2B points
extend the iso-token trend cleanly (still monotonic; 1e21 0.93687 > 1e22 0.85640).

### 1e22 tok2b placement saga (the hard part — read before re-running a 9.7B preemptible cell)
1e22 (9.7B) has a ~15-min HF-load window (40GB from HF @ ~46MB/s) BEFORE its first checkpoint. On
**preemptible** v5p that window keeps getting killed by **GCP spot reclamation** (`Worker ... failed:
worker ping threshold exceeded` — one of the 4 coscheduled spot VMs vanishes, bouncing the whole slice;
NOT an iris priority-tenant preemption). Sequence:
- **a001 v5p-32 us-east5-a**: bound, preempted mid-load repeatedly, 0 steps. The 2 "ready" v5p-32 slices
  were CPU-tenant-blocked (unbindable), so every retry waited on a fresh spot boot that then got reclaimed.
- **a002 v4-reserved-128 us-central2-b** (moved for reserved/non-preempt + 64 chips ~7h): the fresh 16-host
  reserved slice **wedged in GCP provisioning** (`Worker not found` 29min, never came up). us-central2-b
  reserved v4 was degraded that day (also stranded the 1e21 tok500m straggler ~hours). Killed.
  Added `v4-128` to the 1e22 ALLOWED_TPUS_PER_BASE for this (one-line, deliberate).
- **a003 v5p-32 us-east5-a** (reverted): a 4-host v5p-32 binds far more reliably than a 16-host v4-128.
  Thrashed (preemptions=2, 0 steps) while contention was high, then **caught a clean window once the
  overnight `/held`+`/larry` load eased**, loaded, hit first checkpoint, and self-healed from temp ckpt
  through later spot churn -> ran to 477. **This is the lesson: for a 9.7B preemptible CPT cell, the only
  vulnerable moment is the first load; once past step-1 checkpoint, spot reclamation is cheap. Reserved
  (non-preempt) capacity avoids it entirely but the only reserved TPU is v4 us-central2-b (no reserved v5p).**
- Confirmed: v5p worker schedulable RAM = 432 GiB (bug-report) so `--ram 400g` always fit; RAM was never
  the blocker (the "16GB node" the DRI saw was the n2-highmem-2 **coordinator**, which only needs 3GB).

### State / next
- **All targets done:** tok500m 9/9, tok2b 1e21+1e22. No jobs running.
- 1e22 tok2b is in us-east5 (`gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-tok2b-lr50-a003`); 1e21
  tok2b us-east5; tok500m all us-central2.
- **Still TODO for full seal:** rerun `scripts/analysis/delphi_isotoken_endpoint_scaling.py` (it auto-picks
  tok500m/tok1b/tok2b by tag) for the updated overlay + fit-error report across all three budgets.
- Uncommitted launcher change this round: `v4-128` added to `ALLOWED_TPUS_PER_BASE["1e22"]` (commit with
  the rest of the uncommitted launcher work if keeping).

### Endpoint-scaling rerun (all budgets) + error-vs-tokens
Reran `scripts/analysis/delphi_isotoken_endpoint_scaling.py --no-history` (picks up tok500m/tok1b/tok2b by
tag; cache in `midtrain_wandb_data/`). Held-out extrapolation error, **Chinchilla floor+power fit through
3e20**, mix p33m67 lr0.5 (pred/actual − 1):
| ladder | 1e21 | 1e22 |
|---|---|---|
| iso-FLOP K=0.20 (D grows w/ N) | +2.92% | **+18.57%** |
| iso-token tok500m | −2.31% | −3.66% |
| iso-token tok1b | −2.17% | −3.78% |
**The iso-token error is small and ~flat across 500M→1B (−2..−4%), and the SIGN flips** (iso-FLOP
over-predicts = the "miss"/acceleration; iso-token slightly under-predicts = mild floor). Confirms the
"1e22 miss" is the iso-FLOP token-budget confound, robustly, at every fixed budget tested.
- **tok2b has NO extrapolation-error point**: only 1e21+1e22 were run at 2B (DRI-directed), so there is no
  3e18→3e20 small ladder at 2B to fit-and-extrapolate from. The error-vs-tokens plot
  (`sk_midtrain_analysis_fable/error_vs_tokens.png`) therefore shows 500M + 1B only, vs the iso-FLOP line.
- Tooling: installed `matplotlib` into `.venv` via `uv pip install` (the repo's analysis stack is Plotly /
  no kaleido) to render the static PNG; the interactive overlay HTML is still emitted as before.

### NEXT (queued for /goal): run the 2B small ladder to complete the error-vs-tokens curve
Run the **7 small-ladder bases at 2B** — 3e18, 9e18, 2e19, 3e19, 9e19, 2e20, 3e20 — mix p33m67, lr0.5,
`--budget-tokens 2000000000`. CPT loads from HF → region-free → use ANY free compute (v6e-4/v4/v5p,
preemptible or reserved, any region). Per-base 2B step counts (= 2× the 1B table): 3e18 61036, 9e18 30518,
2e19 30518, 3e19 15258, 9e19 7629, 2e20 7629, 3e20 3814. Small batches/models → small slices fine; no int32
risk at these scales (overflow was a 1e22-only issue). Cross-region (non us-east5) needs `--region <R>` +
`-e MARIN_I_WILL_PAY_FOR_ALL_FEES 1`. On completion: rerun the endpoint script + regenerate
`error_vs_tokens.png` so it gains its 2B point (true 500M→1B→2B error trend).

### SEALED — tok2b small ladder (7 bases) DONE → full 9-base tok2b ladder complete
Ran the 7 small-ladder bases at 2B (mix p33m67, lr0.5), all in-region us-east5: 3e18→v5p-16 init
(later v6e-8), 9e18/2e19/3e19/9e19→v6e-8 us-east5-b, 2e20/3e20→v5p-8 (later v5p-32). Batch held canonical
per base (verified: steps×batch×4096 = 2e9). Final tok2b 4plus/loss (best finished attempt per base):
| 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---|---|---|---|---|---|---|---|
|1.19127|1.14841|1.09131|1.07055|1.02569|0.98944|0.96681|0.93687|**0.85640**|
**Strictly monotonic in scale; no crossover** — same clean shape as tok500m/tok1b. Per-base token trend
(500M>1B>2B) holds everywhere, e.g. 3e20 1.09256→1.03000→0.96681, 9e19 1.15586→1.08989→1.02569.

### Error-vs-tokens — the headline result (Chinchilla floor+power, fit through 3e20)
Held-out extrapolation error (pred/actual−1) at the two large scales, now across ALL three iso-token budgets
plus the iso-FLOP reference:
| ladder | 1e21 | 1e22 |
|---|---|---|
| iso-FLOP K=0.20 (D grows w/ N) | +2.92% | **+18.57%** |
| iso-token tok500m | −2.31% | −3.66% |
| iso-token tok1b | −2.17% | −3.78% |
| iso-token tok2b | −2.12% | −3.59% |
**The iso-token error is flat & small (~−2 to −4%) across the entire 4× token range (500M→1B→2B), and the
sign is flipped vs iso-FLOP.** So the "1e22 miss" (+18.6% over-prediction on iso-FLOP) is a token-budget
confound of the iso-FLOP ladder (D grows with N), NOT a 1e22 effect — and that conclusion is now robust to
the midtraining token budget, not a single-budget artifact. Plot: `sk_midtrain_analysis_fable/error_vs_tokens.png`
(matplotlib, the 500M→1B→2B curve); interactive overlay HTML regenerated alongside.

### Ops: spot-preemption wave + the resume fix (new launcher capability)
~8.5h in, a single GCP spot-reclamation wave hit all 3 of my v5p-us-east5-a cells at once (3e18 on v5p-8,
2e20 on v5p-8, 3e20 on v5p-16); the small v5p pools then drained to 0 ready (while v5p-32 sat at 22 idle),
so the preempted tasks couldn't re-allocate and sat stuck ~30+ min (v6e-8 cells were unaffected). Fix:
- The launcher REFUSES a fresh CPT launch when checkpoints already exist (anti-clobber guard in
  `preflight.py`), pointing to `expected_min_step` to resume. That field existed on `MidtrainSpec` but was
  not exposed — **added `--expected-min-step N`** to the launcher (threads to `build_spec`→`MidtrainSpec`).
  Setting it makes preflight verify a checkpoint ≥ N exists, then training resumes from the latest GCS ckpt
  (+ W&B step counter) instead of cold-starting from HF. **This is the canonical "move a preempted cell to
  fresh capacity without losing progress" lever** (reuse the same `--attempt`; give the iris `--job-name` a
  fresh timestamp). NOTE: a *new* `--attempt` still cold-starts (different output path) — that's why the 1e22
  1B saga's a004/a006 redid from scratch; reuse the SAME attempt + `--expected-min-step` to resume.
- Killed the 3 stuck jobs; relaunched `--attempt 1 --expected-min-step <≈last ckpt>` onto idle in-region
  capacity: 3e18→**v6e-8** (added to its allowlist; batch 8→per-dev 1), 2e20+3e20→**v5p-32**. All 3 resumed
  from temp ckpts right at their pre-preemption steps (temp ckpts every ~3 steps ⇒ ~zero redo) and ran to
  completion. Verify a resume by FRESH heartbeat at the ckpt step (a cold-start would climb from ~0).

### Uncommitted launcher changes (cumulative this investigation, lint TBD)
`experiments/midtrain_specs/delphi_small_cpt_k020.py`: `--budget-tokens`, `--per-device-parallelism`,
`--region`+cross-region opt-in, `--no-child-preempt`, **`--expected-min-step` (resume)**; allowlist adds
`v4-128`→1e22 and `v6e-8`→3e18; `default_budget_label` → `500m`/`2b`. Plus `spec.py`/`preflight.py`
cross-region opt-in. Commit the lot if keeping. Analysis: `scripts/analysis/delphi_isotoken_endpoint_scaling.py`
(+ matplotlib installed in `.venv` via `uv pip` for `error_vs_tokens.png`).

## State — iso-token study COMPLETE
All three budgets done across the ladder: **tok500m 9/9, tok1b 9/9, tok2b 9/9** (tok2b = the 7 small bases
this session + 1e21/1e22 from the prior session). No jobs running. The crossover/"1e22 miss" question is
fully answered: it is an iso-FLOP token-budget confound; at fixed D the N-scaling extrapolation error is
small, stable, and sign-flipped at every budget tested. Remaining open experiments from the original plan
(#1 decontaminated re-eval, #2–#6) are unaffected and still available on these checkpoints.

### Chinchilla vs log-log: both fits agree; how the error moves 500M→2B
Full held-out error table (fit through 3e20; signed pred/actual−1), both scaling forms:
| budget | 1e21 Chin | 1e21 log-log | 1e22 Chin | 1e22 log-log |
|---|---|---|---|---|
| 500M | −2.31% | −2.49% | −3.66% | −4.27% |
| 1B | −2.17% | −2.23% | −3.78% | −3.97% |
| 2B | −2.12% | −2.12% | −3.59% | −3.60% |
| iso-FLOP | +2.92% | +2.11% | +18.57% | +15.34% |
Observations: (1) **both fits agree** on a small NEGATIVE iso-token error at every budget (model mildly beats
the extrapolation = floor/flattening), so the contrast with iso-FLOP's large POSITIVE miss is robust to the
scaling form, not a Chinchilla-vs-log-log artifact. (2) **Trend 500M→2B:** errors do NOT grow with tokens —
1e21 shrinks monotonically toward 0 (both fits); 1e22 log-log shrinks (−4.27→−3.60), 1e22 Chinchilla ~flat
(−3.66→−3.78→−3.59). (3) **The two fits CONVERGE at 2B** (1e21 both −2.12; 1e22 −3.59 vs −3.60) because the
Chinchilla floor E collapses to 0 with more tokens (E = 0.1942→0.0650→**0.0000** at 500M/1B/2B) — at 2B the
floor+power IS a pure power law = the log-log form. (4) iso-token log-log slope ≈ −0.045 stable across budgets
vs iso-FLOP −0.098 (half the apparent iso-FLOP "scaling" is the growing D). Plots:
`sk_midtrain_analysis_fable/error_vs_tokens.png` (500M→2B curve) + `err_fit_compare.png` (Chinchilla vs log-log,
per held-out scale).

## 2026-06-15 — /goal: tok4b 9-base ladder + 1e22 tok8b (extend the error-vs-tokens curve)
Launched the full 9-base ladder at **4B** (`--budget-tokens 4000000000`, mix p33m67, lr0.5), batch held
canonical per base (verified steps = 2× the 2B table). Per-base tok4b step targets: 3e18 122070, 9e18 61035,
2e19 61035, 3e19 30518, 9e19 15259, 2e20 15259, 3e20 7629, 1e21 1907, 1e22 954.

**Status (2026-06-16 ~06:30Z): 4/9 DONE, 5 running healthy.**
| base | step/target | state | 4plus/loss |
|---|---|---|---|
| 3e18 | 114399/122070 (~94%) | running | 1.1323 (sofar) |
| 9e18 | 61034/61035 | **DONE** | 1.08628 |
| 2e19 | 18762/61035 (~31%) | running ← long pole | 1.1860 (sofar) |
| 3e19 | 14799/30518 (~48%) | running | 1.1042 (sofar) |
| 9e19 | 15258/15259 | **DONE** | 0.96310 |
| 2e20 | 9999/15259 (~66%) | running | 0.97674 (sofar) |
| 3e20 | 7628/7629 | **DONE** | 0.90277 |
| 1e21 | ~1080/1907 (~57%) | running | 0.93820 (sofar) |
| 1e22 | 953/954 | **DONE** | **0.79481** |
4B per-base token trend continues 2B→4B (e.g. 1e22 0.85640→0.79481; 9e19 1.02569→0.96310; 3e20 0.96681→0.90277).
Slices: 2e19/3e19 on v6e-8 us-east5-b (`...-v6eback-...`); 1e21/2e20 on v5p us-east5; 3e18 on v5p-8 us-east5.

### 1e22 tok8b — relocated v5p-128 → v5p-32 us-east5 (resume from temp step-41)
Launched 1e22 at **8B** (1907 steps, batch 1024 unchanged) on **v5p-128** ("while we can"). It cleared the
9.7B HF cold-load (reached step 56) but one worker hit SIGSEGV (exit 139); the gang then could not re-place:
**v5p-128 went to 0 ready / 0 booting / 0 demand in BOTH us-east5 AND us-central1** (cluster status), gang
stuck pending >1h. The big-slice window closed. Fix-and-refire: killed the v5p-128 coordinator and relaunched
on **v5p-32 us-east5** (13 ready) — same region (no cross-region read), HBM fits at pdev16, batch stays 1024
(4× grad-accum vs v5p-128). Resumed **same attempt a001 from the step-41 temp checkpoint** via
`--expected-min-step 41` (preflight `_latest_step_across` reads temp ckpts too), so no second HF cold-load.
Coordinator `delphi-iso-1e22-tok8b-v5p32-20260615-232739`; child = 4-task v5p-32 gang, pending placement.
~4× slower per step than v5p-128 but actually progresses. Lesson reaffirmed: only relocate UNPLACED gangs to
chase capacity — never kill a placed/working job for a speculative speedup.

### 2026-06-16 ~15:30Z — 4h silent stall of the 3 remaining 4B/8b jobs + recovery (two bugs)
After 1e21(4B) finished (0.87504; trend 1.06156→0.99927→0.93687→0.87504), the 3 still-active runs
(2e19, 3e19 on v6e-8 us-east5-b; 1e22 tok8b on v5p-32 us-east5-a) **all froze simultaneously and sat dead
~3.9h** before I caught it. Root cause = a host/infra event that killed all three CPU coordinators at once:
- **3e19 coordinator**: `worker_lost_spec` → state `killed` (terminal, iris does NOT auto-retry a lost host).
- **1e22 tok8b coordinator**: gone (`not found`).
- **2e19 coordinator**: `failed` exit 1. **Latent bug**: the 4B coordinators were launched WITHOUT
  `--expected-min-step`, so when iris retried the coordinator after the host blip, the launcher re-ran a
  *fresh* `--attempt 2` launch, the anti-clobber guard saw the now-existing checkpoints, and refused
  (exit 1). A deterministic failure → retries exhausted → terminal.
Children cascade-died with their coordinators ⇒ W&B heartbeats went stale but steps never advanced.
**Why it went unnoticed 4h**: my watcher only fired on *finishes*, and one poll-loop had died earlier on a
transient W&B ConnectTimeout. **Fixes applied:**
1. **Resume-safe relaunch**: killed/relaunched all 3 with `--expected-min-step` (2e19 @30515 perm, 3e19 @23601
   temp, tok8b @190 perm). This both resumes from checkpoint AND makes the coordinator idempotent under future
   iris retries (retry re-runs the same cmd → now a resume, not a clobber-refuse). **Always pass
   `--expected-min-step` on any long coordinator that can be restarted.**
2. **Zone bug**: first 2e19/3e19 relaunch used the generic wrapper's `--zone us-east5-a`, but **v6e-8 lives in
   us-east5-b**; the coordinator `--zone` propagates to the child ⇒ "unschedulable: no groups in zone
   us-east5-a; did you mean us-east5-b?". Re-relaunched 2e19/3e19 with `--zone us-east5-b`. (v5p-32 tok8b
   stays us-east5-a — that's where v5p is.) **v6e-8 ⇒ us-east5-b; v5p ⇒ us-east5-a.**
3. **Stall detection** added to the watcher (`/tmp/watch_stall.py`): any non-DONE run with W&B heartbeat
   >20min for 2 consecutive polls ⇒ `EVENT_STALL` (with a ~20min warmup so just-resumed runs aren't flagged
   mid-recompile). Detects a freeze in ~25min instead of 4h.
Recovery coordinators: `delphi-iso-{2e19,3e19}-tok4b-resb-20260616-082838` (us-east5-b),
`delphi-iso-1e22-tok8b-v5p32r-20260616-082450` (us-east5-a). 4B ladder 7/9 done; ETA ~12h gated on 2e19.

### 2026-06-16 ~18:00Z — /goal extension: full 8B ladder launched (all 9 bases)
DRI chose the FULL 8B ladder (apples-to-apples 8B point on error-vs-tokens, same fit anchors as 4B). 8B step
targets = 2× the 4B table: 3e18 244140, 9e18 122070, 2e19 122070, 3e19 61036, 9e19 30518, 2e20 30518,
3e20 15258, 1e21 3814, 1e22 1908. **The two giants 3e18 (~3-4d) and 9e18 (~2d) dominate wall-clock** (tiny
batches → huge step counts). Launched 7 fresh cold-start cells (attempt 1, from HF), all v5p us-east5-a to
dodge the us-east5-b CPU-coordinator shortage: 3e18/9e18/3e19 → v5p-8; 9e19/2e20/3e20 → v5p-32; 1e21 → v5p-32
pdev16 (replicating 4B per-base pdev). 1e22 8B already on v5p-128 (~59%, flying). 2e19 8B deferred until its
4B finishes (~74% now). Coordinators `delphi-iso-{base}-tok8b-20260616-175415`.

Confirmed mechanism (matters for the multi-day giants): a fresh attempt-1 launch self-heals through normal
iris coordinator preemption-retries — `_is_iris_retry_attempt()` returns True (`attempt_id>0`) and preflight
bypasses the anti-clobber guard for a matching-manifest retry (preflight.py:295-301), so the retry resumes
from Levanter's latest checkpoint instead of refusing. Only a TERMINAL coordinator death (host worker_lost →
killed) needs a manual relaunch with `--expected-min-step`; the stall watcher (`/tmp/watch_8b.py`, >22min
heartbeat staleness) catches that in ~25min. (Corrects the earlier guess that the clobber guard killed 2e19 —
the guard is bypassed on retries; 2e19's coordinator died for another reason during the host event.)

## 2026-06-17 — SEALED: 4B ladder COMPLETE (9/9) + error-vs-tokens extended to 4B
All nine 4B cells finished. Final nemotron_cc_math_v1/4plus loss (best attempt per base), strictly monotonic
in scale, NO crossover:
| 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---|---|---|---|---|---|---|---|
|1.12541|1.08628|1.02741|1.00650|0.96310|0.92644|0.90277|0.87482|**0.79481**|

Endpoint-scaling rerun (fit through 3e20, n=7) adds the 4B held-out error point — it lands right on the flat
band, headline holds:
| budget | 1e21 Chin | 1e21 log-log | 1e22 Chin | 1e22 log-log |
|---|---|---|---|---|
| 500M | −2.31% | −2.49% | −3.66% | −4.27% |
| 1B | −2.17% | −2.23% | −3.78% | −3.97% |
| 2B | −2.12% | −2.12% | −3.59% | −3.60% |
| **4B** | **−2.28%** | **−2.30%** | **−3.73%** | **−3.77%** |
| iso-FLOP K=0.20 | +2.92% | +2.11% | **+18.57%** | +15.34% |
**iso-token extrapolation error is flat & small (1e21 ~−2.3%, 1e22 ~−3.7%) across the full 8× token range
(500M→4B), sign-flipped vs the iso-FLOP +18.6% "1e22 miss" — which is therefore an iso-FLOP token-budget
confound (D grows with N), robust to the midtraining budget.** 4B fit: floor=0.0000 amp=1.1924 alpha=0.0482
(R²=0.99255); log-log slope −0.0482 (vs iso-FLOP −0.0977 — half the apparent iso-FLOP "scaling" is growing D).
Plot regenerated: `sk_midtrain_analysis_fable/error_vs_tokens.png` (500M→1B→2B→4B curve) + interactive HTML.

### 8B ladder — IN FLIGHT (all 9 launched)
Per DRI: full 8B ladder (apples-to-apples 8B error point). 1e22 8B DONE = **0.72037** (trend
0.97412→0.91762→0.85640→0.79481→**0.72037**, clean across 16× tokens). Other 8 launched; once 2e19-4B finished
its 8B (122k steps) launched too. Ops reality logged below: us-east5 spot is volatile — single-worker v5p-8
places reliably (giants 3e18/9e18/2e19 live there); multi-worker v5p-32/64 gangs are flaky and hang silently
(iris reports "running" while the process is wedged → NO auto-retry). A ~1h capacity crunch (2026-06-16 ~02-03Z
UTC) left everything pending/hung; it resolved on its own and iris re-placed all pending jobs — **lesson: in a
crunch, stop relaunching (futile churn) and let iris's pending-retry self-heal; only manually relaunch the
truly HUNG (worker-log silent >15min, not the W&B summary _timestamp which only refreshes at eval).** Giants
(3e18 244k, 9e18 122k, 2e19 122k steps) are a multi-day tail. Monitor: `/tmp/watch_8bonly.py` (report-only,
fires on all-9-8B-done). 8B targets = 2× the 4B step counts.

## 2026-06-18 — SEALED: 8B ladder COMPLETE (9/9) — full iso-token study done (500M→8B, 16×)
All nine 8B cells finished. Final nemotron_cc_math_v1/4plus loss, strictly monotonic in scale, NO crossover:
| 3e18 | 9e18 | 2e19 | 3e19 | 9e19 | 2e20 | 3e20 | 1e21 | 1e22 |
|---|---|---|---|---|---|---|---|---|
|1.06714|1.02996|0.96791|0.94596|0.90147|0.86239|0.83573|0.80592|**0.72012**|

Endpoint-scaling rerun adds the 8B held-out error point (fit through 3e20, n=7): floor=0.0000 amp=1.1400
alpha=0.0533 (R²=0.99038); log-log slope −0.0535. The 8B point lands in the flat band and the 1e22 error
actually SHRINKS:
| budget | 1e21 Chin | 1e21 log-log | 1e22 Chin | 1e22 log-log |
|---|---|---|---|---|
| 500M | −2.31% | −2.49% | −3.66% | −4.27% |
| 1B | −2.17% | −2.23% | −3.78% | −3.97% |
| 2B | −2.12% | −2.12% | −3.59% | −3.60% |
| 4B | −2.28% | −2.30% | −3.73% | −3.77% |
| **8B** | **−2.08%** | **−2.16%** | **−3.07%** | **−3.19%** |
| iso-FLOP K=0.20 | +2.92% | +2.11% | **+18.57%** | +15.34% |
**FINAL RESULT: the iso-token N-scaling extrapolation error is flat & small (1e21 ~−2.1%, 1e22 ~−3.1%) across
the ENTIRE 500M→8B 16× token sweep, sign-flipped vs the iso-FLOP +18.6% "1e22 miss" — which is therefore an
iso-FLOP token-budget confound (D grows with N on the iso-FLOP ladder), conclusively NOT a 1e22 effect and
robust across a 16× midtraining-token range.** Per-base token trend holds everywhere (e.g. 1e22
0.97412→0.91762→0.85640→0.79481→0.72012 across 500M→8B). Plot: `sk_midtrain_analysis_fable/error_vs_tokens.png`
(500M→8B curve) + interactive HTML.

### 8B ops notes (multi-day grind on us-east5 spot)
- Single-worker v5p-8 = stable; multi-worker gangs (v5p-16/32/64) coschedule + hang more (silent: iris
  "running" while wedged → no auto-retry; detect via worker-log silence >15min, NOT W&B summary _timestamp
  which only refreshes at eval). Recurring host-blip hangs cleared by kill+relaunch-from-temp-checkpoint.
- **Speedup move (DRI request):** moved the three giants off v5p-8 to bigger v5p (resume-from-checkpoint, same
  us-east5 bucket, no copy): 3e18→v5p-16 (batch 8 caps data-parallel at 8 chips), 9e18+2e19→v5p-32 (batch 16
  → 16 chips). Measured ~1.35× (3e18), ~1.7× (9e18), ~1.9× (2e19) — sublinear (tiny batches are overhead-bound)
  but halved the long-pole ETA (~38h→~20h). Added v5p-32 to 9e18/2e19 allowlists for this.
- Lesson reaffirmed: in a us-east5 capacity crunch, STOP relaunching (futile thundering-herd churn) — iris's
  pending-retry self-heals every preempted job once capacity returns; only manually relaunch the truly HUNG.

### 2026-06-18 — Definitive plot: the "1e22 miss" mechanism, side by side
`sk_midtrain_analysis_fable/isoflop_miss_is_token_artifact.png` — the single clearest figure of the result.
Two panels, log-log val-loss vs pretrain-FLOP scale C, SAME extrapolation procedure (Chinchilla floor+power
fit through 3e18–3e20, then predict the held-out 1e21/1e22):
- **LEFT, iso-FLOP K=0.20 (D grows with N, 0.24B→32B):** 1e22's actual loss (0.56102) sits FAR BELOW the fit's
  prediction (0.66520) → **+18.58%** — the "miss". 1e22 looks anomalously good only because it also got vastly
  more midtraining tokens (32B) than the small ladder, which the N-only fit can't see.
- **RIGHT, iso-token (fixed D = 8B every base):** the same 1e22 model lands ON the fit (pred 0.69805 vs actual
  0.72012 = **−3.11%**), and 1e21 is **−2.08%**.
Hold the token budget fixed and the +18.6% miss collapses to ~−3% → **the miss is a token-budget artifact of
the iso-FLOP ladder, not a 1e22-scale effect.** Per-base iso-FLOP losses used (k0p20 small + 9p25b/32p07b
large): 3e18 1.43522, 9e18 1.27370, 2e19 1.19341, 3e19 1.13897, 9e19 1.02104, 2e20 0.95401, 3e20 0.90956,
1e21 0.79353, 1e22 0.56102. Generator: `/tmp/definitive_plot.py` (data fetched live from W&B; fit params from
the endpoint-scaling print_summary). Companion: `error_vs_tokens.png` (the −2…−4% flat band across 500M→8B).

# ========================================================================
# HANDOFF FOR NEXT AGENT — 2026-06-18 (iso-token CPT study: COMPLETE)
# ========================================================================

## TL;DR — the study is DONE and the question is answered
The iso-token midtraining CPT ladder is complete across **5 token budgets × 9 model scales**
(500M, 1B, 2B, 4B, 8B × 3e18…1e22), mix p33m67, lr0.5. **No jobs are running.** The original question —
is the iso-FLOP "1e22 miss" (+18.6% scaling-law over-prediction) real? — is conclusively answered:

**It is an iso-FLOP token-budget artifact, NOT a 1e22 effect.** On the iso-FLOP ladder the midtrain token
budget D grows with model scale N (0.24B→32B), so the N-only scaling fit can't see the extra tokens and
1e22 looks anomalously good (+18.6% below the fit). Hold D fixed (iso-token) and the same 1e22 lands ON the
fit (−3.1%). This holds at EVERY budget across the full 16× token sweep (1e21 ~−2.1%, 1e22 ~−3.1%).

## Final numbers (nemotron_cc_math_v1/4plus val loss; metric key `eval/nemotron_cc_math_v1/4plus/loss`)
Both ladders strictly monotonic in scale, no crossover.
- tok4B: 3e18 1.12541 / 9e18 1.08628 / 2e19 1.02741 / 3e19 1.00650 / 9e19 0.96310 / 2e20 0.92644 /
  3e20 0.90277 / 1e21 0.87482 / 1e22 0.79481
- tok8B: 3e18 1.06714 / 9e18 1.02996 / 2e19 0.96791 / 3e19 0.94596 / 9e19 0.90147 / 2e20 0.86239 /
  3e20 0.83573 / 1e21 0.80592 / 1e22 0.72012
- Held-out error (Chinchilla floor+power fit through 3e20): 1e21 ranges −2.08…−2.31%, 1e22 −3.07…−3.78%
  across 500M→8B; iso-FLOP K=0.20 is 1e21 +2.92%, **1e22 +18.57%**. Full table is in the
  "2026-06-18 SEALED" entry above.

## Deliverables / artifacts
- **Plots** (in `sk_midtrain_analysis_fable/`, which is GITIGNORED):
  - `isoflop_miss_is_token_artifact.png` — the money figure (2-panel scaling fit: 1e22 off the iso-FLOP fit,
    on the iso-token fit). Generator: **`scripts/analysis/plot_isoflop_miss_token_artifact.py`** (fetches data
    live from W&B; fit params hardcoded from print_summary).
  - `error_vs_tokens.png` — the flat −2…−4% error band across 500M→8B (categorical x-axis, no log-tick
    garbage). Generator: **`scripts/analysis/plot_isotoken_error_vs_tokens.py`** (error values hardcoded from
    print_summary). Both generators run from the repo root and write into `sk_midtrain_analysis_fable/`.
  - `delphi_isotoken_endpoint_scaling.html` — interactive overlay (regenerated by the analysis script).
- **Analysis script**: `scripts/analysis/delphi_isotoken_endpoint_scaling.py` — rerun to refresh fits +
  print_summary (the per-budget held-out error table). Slow (~5min, queries W&B); run WITHOUT a `timeout`
  wrapper or in background. iso-FLOP runs are NOT under the sweep tag — they match `*-k0p20-*` (small) and
  `*-9p25b-*`/`*-32p07b-*` (1e21/1e22 large) and live in the local cache `midtrain_wandb_data/runs/`.

## ⚠️ UNCOMMITTED — needs a commit (offered to DRI, not yet greenlit)
The entire investigation is uncommitted on branch `midtrain_data`:
- `experiments/midtrain_specs/delphi_small_cpt_k020.py`: `--expected-min-step` resume flag, `--budget-tokens`,
  `--per-device-parallelism`, `--region`/cross-region opt-in, `--no-child-preempt`; allowlist widenings
  (v4-128→1e22, v6e-8→3e18, **v5p-32→9e18/2e19** added this session for the speedup move).
- `lib/marin/src/marin/midtraining/budget.py` (`default_budget_label` → 500m/2b/4b/8b).
- `scripts/analysis/delphi_isotoken_endpoint_scaling.py`, this logbook, `.agents/logbooks/delphi_true_cooldown.md`.
- New (untracked): `scripts/analysis/plot_isotoken_error_vs_tokens.py` + `scripts/analysis/plot_isoflop_miss_token_artifact.py`
  (the two figure generators, preserved out of `/tmp` so they survive; `git add` them with the commit).
When committing: `./infra/pre-commit.py --changed-files --fix`, then commit, then `./infra/pre-commit.py
--review`, then push. (Worktree at `.claude/worktrees/midtrain`; `.marin.yaml` symlinked for WANDB key.)

## Ops lessons (us-east5 spot is hostile to long multi-job runs)
- **Single-worker v5p-8 = stable; multi-worker gangs (v5p-16/32/64) coschedule slowly and HANG silently**
  (iris reports `running` while the process is wedged → no auto-retry). Detect a real hang by **worker-log
  silence >15min**, NOT the W&B summary `_timestamp` (it only refreshes at eval boundaries → false stalls
  between evals). Step-advancement is a better signal than timestamp but still pauses during eval.
- **Resume lever**: relaunch with the SAME `--attempt` + `--expected-min-step <≤latest ckpt>` (find via
  `gsutil ls .../checkpoints/` and the temp path `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/...`).
  Levanter resumes from the latest of perm/temp. A *fresh* launch self-heals through iris coordinator
  preemption-retries (`_is_iris_retry_attempt` → attempt_id>0 bypasses the anti-clobber guard); only a
  TERMINAL coordinator death (host `worker_lost` → killed) needs a manual relaunch.
- **In a capacity crunch, STOP relaunching** — it's a futile thundering herd. iris's pending-retry re-places
  every preempted job once capacity returns; only manually relaunch the truly HUNG.
- **Speedup**: bigger v5p helps but sublinear for tiny batches (overhead-bound): measured ~1.35×/1.7×/1.9×
  for 3e18/9e18/2e19. Data-parallel width ≤ global batch ⇒ 3e18 (batch 8) caps at v5p-16 (8 chips);
  9e18/2e19 (batch 16) cap at v5p-32 (16 chips). **Zone: v6e-8 ⇒ us-east5-b, v5p ⇒ us-east5-a** (coordinator
  `--zone` propagates to the child).

## What's NOT done (available, not started — all on these CPT checkpoints)
The other original experiments (#1 decontaminated re-eval, #2–#6) are untouched and unaffected by this study.

## 2026-06-21 - 2D Chinchilla with base-loss variants

Added `scripts/analysis/delphi_midtraining_2d_chinchilla.py` to test whether a two-resource Chinchilla fit should
be constrained by base-model math validation losses at zero midtraining tokens.

Command:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_fit.json`

Setup:

- Endpoint rows are the p33m67/lr50 iso-token ladder plus K=0.20 iso-FLOP ladder.
- Training split is still 3e18 through 3e20; 1e21 and 1e22 remain held out.
- Base-loss rows come from
  `midtrain_analysis_outputs/small_final_loss_scaling/trajectory_points.csv`, filtered to
  `metric_label == "math_val_loss"` and `step == 0`.
- The HTML dropdown switches among:
  1. endpoints only,
  2. base as D=epsilon with `epsilon=0.05B`,
  3. separate base-loss curve plus midtraining improvement term.

Held-out summary:

| model | n | MAE % | RMSE % | bias % | note |
|---|---:|---:|---:|---:|---|
| endpoints only | 12 | 2.08 | 4.21 | +0.93 | current two-resource fit |
| base as D=epsilon | 14 | 8.55 | 10.03 | -1.60 | worsens endpoint and base heldout fit |
| separate base component | 14 | 6.31 | 9.24 | -0.34 | fits heldout base points well, but overpredicts K=0.20 at 1e22 |
| K=0.20-only Chinchilla | 2 | 10.75 | 13.29 | +10.75 | right-plot baseline only |

Interpretation: adding base losses is a useful constraint check but does not improve the held-out endpoint
errors in this first parameterization. The epsilon-token version is too rigid at D=0 and badly overpredicts
heldout base losses; the separate-base version handles D=0 cleanly but its improvement term extrapolates poorly
to the 32B-token K=0.20 1e22 point.

## 2026-06-21 - Tiny epsilon rerun

Changed the shifted-token base-loss mode from `epsilon=0.05B` to `epsilon=1e-7B`. Since `tokens_b` is in
billions, the previous value was 50M tokens; the new value is about 100 tokens. Also decoupled the
separate-base model from that epsilon: mode 3 now fits its own positive token scale for the improvement term,
while still forcing zero improvement at `D=0`.

Command:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py
```

Updated artifacts:

- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_fit.json`

Held-out summary after the tiny-epsilon rerun:

| model | n | MAE % | RMSE % | bias % |
|---|---:|---:|---:|---:|
| endpoints only | 12 | 2.08 | 4.21 | +0.93 |
| base as D=epsilon (`1e-7B`) | 14 | 10.80 | 15.03 | +6.46 |
| separate base component | 14 | 4.92 | 7.14 | -0.60 |
| K=0.20-only Chinchilla | 2 | 10.75 | 13.29 | +10.75 |

Interpretation: the near-zero shifted-token variant is a poor constraint for this additive inverse-token form.
The separate-base parameterization is less bad and keeps the heldout base points close, but endpoints-only still
has the best heldout endpoint error in this comparison.

## 2026-06-21 - HTML explanation pass

Updated `scripts/analysis/delphi_midtraining_2d_chinchilla.py` so the generated HTML is a small report rather
than a bare Plotly figure. The page now explains:

- what `C`, `D`, `epsilon`, and prediction error mean,
- how to interpret positive versus negative held-out error,
- what each dropdown mode is fitting,
- what the `tok*`, `K=0.20`, and `base` series mean,
- why the K=0.20-only Chinchilla line stays visible on the right plot,
- why modes 2 and 3 have more held-out points than mode 1.

Regenerated `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html` and verified the changed-file
pre-commit checks pass.

## 2026-06-21 - Removed shifted-token mode

Removed the `base as D=epsilon` fit from `scripts/analysis/delphi_midtraining_2d_chinchilla.py` and regenerated
the report. The HTML dropdown now only compares:

- `1) endpoints only`
- `2) separate base component`

The K=0.20-only Chinchilla line remains as the right-panel baseline. Verified that the regenerated HTML/CSV/JSON
outputs no longer contain epsilon-mode references and that `./infra/pre-commit.py --changed-files --fix` passes.

## 2026-06-21 - Added log-log mode pair

Added two log-log variants to `scripts/analysis/delphi_midtraining_2d_chinchilla.py` and regenerated the report.
The HTML dropdown now compares four modes:

- `1) Chinchilla endpoints only`
- `2) Chinchilla separate base component`
- `3) log-log endpoints only`
- `4) log-log separate base component`

Implementation notes:

- The endpoints-only log-log mode fits `log L = a + b log C + c log D`.
- The separate-base log-log mode fits a pure power-law base loss from the `D=0` base rows, then fits
  `log(L_base(C) - L(C,D)) = a + b log C + c log D` on positive-token endpoint rows. The improvement is still
  exactly zero at `D=0`.

Held-out summary after adding log-log modes:

| model | n | MAE % | RMSE % | bias % |
|---|---:|---:|---:|---:|
| Chinchilla endpoints only | 12 | 2.08 | 4.21 | +0.93 |
| Chinchilla separate base component | 14 | 4.92 | 7.14 | -0.60 |
| log-log endpoints only | 12 | 3.70 | 5.16 | -1.22 |
| log-log separate base component | 14 | 4.70 | 6.10 | -2.21 |
| K=0.20-only Chinchilla baseline | 2 | 10.75 | 13.29 | +10.75 |

Interpretation: log-log separate-base is a little better than Chinchilla separate-base on held-out MAE/RMSE, but
the original Chinchilla endpoints-only fit still has the lowest held-out error in this comparison.

## 2026-06-21 - Iso-token-only 2D scaling report

Added separate output stems to `scripts/analysis/delphi_midtraining_2d_chinchilla.py` so the combined report and
the iso-token-only report can coexist. The default output is still:

- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla.html`

Running with `--exclude-isoflop` now writes:

- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_isotoken_only.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_isotoken_only_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_isotoken_only_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_2d_chinchilla_isotoken_only_fit.json`

Command:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py --exclude-isoflop
```

The iso-token-only report excludes the K=0.20 iso-FLOP ladder and the K=0.20-only baseline line. Held-out
summary:

| model | n | MAE % | RMSE % | bias % |
|---|---:|---:|---:|---:|
| Chinchilla endpoints only | 10 | 1.14 | 1.30 | -0.44 |
| Chinchilla separate base component | 12 | 3.40 | 4.18 | -2.03 |
| log-log endpoints only | 10 | 3.05 | 3.53 | -3.00 |
| log-log separate base component | 12 | 4.13 | 5.06 | -3.67 |

Interpretation: on iso-token runs alone, Chinchilla endpoints-only is clearly best. Adding the separate base
component still hurts, and log-log fits are worse than Chinchilla in this split.

## 2026-06-21 - Parameter-aware Chinchilla report

Added `--include-parameter-chinchilla` to `scripts/analysis/delphi_midtraining_2d_chinchilla.py`. The report now
can add a fifth endpoint-only mode:

```text
L(N,C,D)=E + A*N^-alpha + B*C^-beta + G*D^-gamma
```

where `N` is trainable parameters in billions, sourced from the canonical Delphi registry
`experiments.delphi_models.DELPHI_BY_FLOPS_KEY`.

Commands:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py --include-parameter-chinchilla

uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py --exclude-isoflop --include-parameter-chinchilla
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_fit.json`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_isotoken_only.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_isotoken_only_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_isotoken_only_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_chinchilla_isotoken_only_fit.json`

Held-out summary:

| report | baseline Chinchilla endpoints MAE | parameter Chinchilla MAE | parameter Chinchilla RMSE |
|---|---:|---:|---:|
| iso-token plus K=0.20 | 2.08% | 1.99% | 4.13% |
| iso-token-only | 1.14% | 0.93% | 1.21% |

Fit exponents:

| report | params exponent | pretraining FLOPs exponent | midtraining tokens exponent |
|---|---:|---:|---:|
| iso-token plus K=0.20 | 0.340 | 0.131 | 0.140 |
| iso-token-only | 0.352 | 0.131 | 0.137 |

Interpretation: adding parameter count improves held-out MAE slightly in both reports, but this is exploratory.
`N` and `C` are strongly correlated across the Delphi ladder, so the separate parameter and FLOP exponents are not
independently identified by this dataset.

## 2026-06-21 - Parameter/data Chinchilla report

Replaced the optional `N + C + D_mid` Chinchilla mode in
`scripts/analysis/delphi_midtraining_2d_chinchilla.py` with a FLOPs-free parameter/data mode:

```text
L(N,D_pre,D_mid)=E + A*N^-alpha + B*D_pre^-beta + G*D_mid^-gamma
```

where `N` is trainable parameters in billions from `experiments.delphi_models.DELPHI_BY_FLOPS_KEY`,
`D_pre` is pretraining tokens in billions, and `D_mid` is midtraining tokens in billions. The HTML hovers now show
`N`, `D_pre`, `D_mid`, and `D_total`.

Commands:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py --include-parameter-data-chinchilla

uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_2d_chinchilla.py --exclude-isoflop --include-parameter-data-chinchilla
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_fit.json`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_isotoken_only.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_isotoken_only_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_isotoken_only_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_param_data_chinchilla_isotoken_only_fit.json`

Held-out summary:

| report | baseline Chinchilla endpoints MAE | parameter/data Chinchilla MAE | parameter/data Chinchilla RMSE |
|---|---:|---:|---:|
| iso-token plus K=0.20 | 2.08% | 1.99% | 4.14% |
| iso-token-only | 1.14% | 0.93% | 1.23% |

Fit exponents:

| report | params exponent | pretraining-data exponent | midtraining-data exponent |
|---|---:|---:|---:|
| iso-token plus K=0.20 | 0.348 | 0.256 | 0.137 |
| iso-token-only | 0.360 | 0.239 | 0.135 |

Interpretation: decomposing pretraining FLOPs into `N` and `D_pre` gives the same small held-out improvement as
the prior parameter-aware fit, while making the Chinchilla resources explicit. The caveat remains that `N` and
`D_pre` are strongly correlated on this ladder.

## 2026-06-21 - Broad fit-family report with math-token resources

Starting a broader report for Delphi midtraining prediction-error fits. Goal: compare endpoint-only and
separate-base forms over multiple resource choices, including raw midtraining tokens, math midtraining tokens,
pretraining tokens, token ratios, and estimated midtraining compute. The report should keep the 1e21/1e22
heldout split and add explicit 1e22 prediction columns because that is the primary target.

Implemented `scripts/analysis/delphi_midtraining_fit_family_report.py`.

Commands:

```bash
uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_fit_family_report.py

uv run --with scipy --with plotly --with pandas --with wandb \
  python scripts/analysis/delphi_midtraining_fit_family_report.py --exclude-isoflop
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_1e22_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_fit_table.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_fit.json`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only.html`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only_summary.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only_1e22_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only_fit_table.csv`
- `sk_midtrain_analysis_fable/delphi_midtraining_fit_family_report_isotoken_only_fit.json`

Fit families now covered:

- endpoint Chinchilla: `L = E + sum_i A_i x_i^-alpha_i`
- endpoint log-log: `log L = a + sum_i b_i log x_i`
- separate Chinchilla: base curve plus saturating improvement, exactly zero at `D=0`
- separate log-improvement: base curve plus power-law improvement on positive-token endpoints

Resource choices include `C_pre`, `N`, `D_pre`, `D_mid`, `D_math`, `D_total`, `K_mid`, `K_math`,
`D_math/N`, `C_mid`, and `C_mid_math`. The HTML explains each resource, provides a Plotly dropdown over all
successful fit modes, and adds explicit summary columns for the focus 1e22 prediction (`k0p20` in the combined
report, `tok8b` in the iso-token-only report).

Headline results:

| report | best model | heldout endpoint MAE | 1e22 focus pred | 1e22 focus actual | 1e22 focus error |
|---|---|---:|---:|---:|---:|
| iso-token + K=0.20 | `N + D_pre + D_mid` Chinchilla | 1.99% | 0.63847 (`k0p20`) | 0.56102 | +13.81% |
| iso-token + K=0.20 | `N + D_pre + D_math` Chinchilla | 1.99% | 0.63847 (`k0p20`) | 0.56102 | +13.81% |
| iso-token-only | `N + D_pre + D_math` Chinchilla | 0.93% | 0.73745 (`tok8b`) | 0.72012 | +2.41% |

As expected for this p33m67-only slice, `D_math = 0.67 * D_mid`, so `D_math` and `D_mid` variants tie when the
fit has a free amplitude. The math-token feature is still represented explicitly for pooled-mix follow-up.

Verified with `./infra/pre-commit.py --changed-files --fix`.

## 2026-06-28 - K=0.20 seen-partition validation plot

Added a focused plot for the completed K=0.20 lr0.50 seen-partition eval sweep:

- New eval root:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_seen_partition_1e22_k020_lr50`
- Script: `scripts/analysis/plot_k020_seen_partition_scaling.py`
- Plot: `sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling.png`
- Points CSV: `sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_points.csv`
- Fit summary CSV: `sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv`

The plot compares the same K=0.20 lr0.50 ladder across old full 4plus, retained clean-seen, and dropped
contaminated subsets. Fits use the same floor+power Chinchilla form through `3e20` and score `1e21`/`1e22`
as heldout.

1e22 heldout prediction errors:

| target | actual | predicted | error |
|---|---:|---:|---:|
| old full 4plus | 0.561143 | 0.665313 | +18.56% |
| retained clean | 0.824991 | 0.848331 | +2.83% |
| dropped contaminated | 0.665261 | 0.765120 | +15.01% |

Verified with:

```bash
uv run python -m py_compile scripts/analysis/plot_k020_seen_partition_scaling.py
./infra/pre-commit.py --changed-files --fix
```

Follow-up: update the same plot and summary table to report absolute loss-unit error alongside relative percent
error, because the old full 4plus and dropped contaminated targets have similar signed loss error but different
percent errors due to different 1e22 loss denominators.

Follow-up: generated a static old-vs-clean version of the iso-token endpoint plot.

Artifact:

- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_vs_old_endpoint_scaling.png`

Script:

- `scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py`

Visual encoding:

- old 4plus validation: dotted Chinchilla fits and hollow actual markers
- clean-seen validation: solid Chinchilla fits and filled actual markers

The plot uses the trusted clean-seen series only: iso-token 2B, 4B, 8B, and K=0.20. It deliberately omits 500M
and 1B because there is no trusted clean-seen 1e22 row for those budgets yet.

Verified with:

```bash
uv run --with scipy --with plotly --with pandas --with gcsfs --with matplotlib \
  python scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py
uv run --with matplotlib --with scipy --with plotly --with pandas --with gcsfs \
  python -m py_compile scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py
./infra/pre-commit.py --changed-files --fix
```

## 2026-06-21 - Add corrected iso-token 1B clean-seen row and simplify static plot

The corrected iso-token 1B clean-seen eval finished:

- Summary CSV:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv`
- Summary JSON:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.json`
- Corrected 1e22/1B row:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/delphi-1e22-p33m67-tok1b-lr50-a008/step-237/metrics.jsonl/eval_results.json`

Plan:

- Include 1B in the unified old-vs-clean report.
- Regenerate sidecar CSV/JSON/HTML artifacts.
- Replace the noisy static overlay with a cleaner primary PNG that emphasizes 1e22 fit error and 1e22 old-vs-clean
  loss, while preserving the old-vs-clean marker semantics.

Outcome: updated `scripts/analysis/delphi_isotoken_clean_seen_unified_report.py` so 1B/2B/4B/8B are all trusted.
Regenerated:

- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_points.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_fit_summary.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.json`

Also replaced `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_vs_old_endpoint_scaling.png` with a cleaner
two-panel 1e22 summary:

- left: 1e22 heldout prediction error after fitting through `3e20`
- right: 1e22 actual validation loss by token budget
- old 4plus: dotted + hollow
- clean-seen: solid + filled

1e22 actual losses:

| series | old 4plus | clean-seen | clean - old |
|---|---:|---:|---:|
| iso-token 1B | 0.917658 | 0.996622 | +0.078963 |
| iso-token 2B | 0.856535 | 0.958299 | +0.101764 |
| iso-token 4B | 0.794952 | 0.926756 | +0.131804 |
| iso-token 8B | 0.720246 | 0.894756 | +0.174510 |
| K=0.20 iso-FLOP | 0.561143 | 0.824991 | +0.263848 |

1e22 fit errors:

| series | old 4plus error | clean-seen error |
|---|---:|---:|
| iso-token 1B | -3.75% | -2.38% |
| iso-token 2B | -3.59% | -2.31% |
| iso-token 4B | -3.72% | -2.74% |
| iso-token 8B | -3.06% | -2.82% |
| K=0.20 iso-FLOP | +18.56% | +2.83% |

Verified with:

```bash
uv run --with scipy --with plotly --with pandas --with gcsfs \
  python scripts/analysis/delphi_isotoken_clean_seen_unified_report.py
uv run --with scipy --with plotly --with pandas --with gcsfs --with matplotlib \
  python scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py
uv run --with matplotlib --with scipy --with plotly --with pandas --with gcsfs \
  python -m py_compile scripts/analysis/delphi_isotoken_clean_seen_unified_report.py \
    scripts/analysis/plot_isotoken_clean_seen_vs_old_endpoint_scaling.py
./infra/pre-commit.py --changed-files --fix
```

## 2026-06-21 - Unified iso-token clean-seen validation report

Built a focused report comparing the old 4plus validation target against the new clean-seen target for the
trusted p33m67/lr0.50 iso-token clean-seen reruns.

Inputs:

- Iso-token clean-seen summary:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv`
- K=0.20 clean-seen summary:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv`
- Verified 1e22 iso-token per-run JSONs for 2B, 4B, and 8B.

Excluded the current 1B/1e22 iso-token row because it is `complete_by_step=False` from
`delphi-1e22-p33m67-tok1b-lr50-a004` at step 50. The corrected a008/step-237 eval should be used once it is
rerun and the summary is regenerated.

Artifacts:

- `scripts/analysis/delphi_isotoken_clean_seen_unified_report.py`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_points.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_fit_summary.csv`
- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.json`

1e22 actual losses:

| series | old 4plus | clean-seen | clean - old | clean/old - 1 |
|---|---:|---:|---:|---:|
| iso-token 2B | 0.856535 | 0.958299 | +0.101764 | +11.88% |
| iso-token 4B | 0.794952 | 0.926756 | +0.131804 | +16.58% |
| iso-token 8B | 0.720246 | 0.894756 | +0.174510 | +24.23% |
| K=0.20 iso-FLOP | 0.561143 | 0.824991 | +0.263848 | +47.02% |

1e22 Chinchilla fit errors when fitting through `3e20`:

| target | iso-token 2B | iso-token 4B | iso-token 8B | K=0.20 iso-FLOP |
|---|---:|---:|---:|---:|
| old 4plus | -3.59% | -3.72% | -3.06% | +18.56% |
| clean-seen | -2.31% | -2.74% | -2.82% | +2.83% |

The new HTML only exposes the two intended targets: `old_4plus_loss` and `clean_seen_loss`. It does not contain
`eval_loss`, `macro_loss`, old eval, or old macro labels.

Verified with:

```bash
uv run --with scipy --with plotly --with pandas --with gcsfs \
  python scripts/analysis/delphi_isotoken_clean_seen_unified_report.py
python -m py_compile scripts/analysis/delphi_isotoken_clean_seen_unified_report.py
./infra/pre-commit.py --changed-files --fix
```

## 2026-06-21 - Unified iso-token old-vs-clean validation report

Starting a unified report that extends the K=0.20 old-vs-clean comparison with the corrected 1e22 iso-token
clean-seen evals for p33m67/lr0.50 at 2B, 4B, and 8B tokens.

Inputs:

- Existing iso-token artifacts in `sk_midtrain_analysis_fable/`, especially `delphi_isotoken_all_budgets_vs_isoflop.png`
  and the cached old-val endpoint CSVs.
- New clean-seen iso-token summary:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_isotoken_p33m67_lr50/summary_p33m67_isotoken_clean_seen_1e22.csv`
- Per-run clean-seen eval result files for 1e22 tok2b/tok4b/tok8b, if the summary needs validation.

Caveat from the eval handoff: ignore the current 1B row in the clean-seen iso-token summary until the corrected
a008/step-237 eval is rerun.

Planned artifact:

- `sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report.html`

## 2026-06-21 - K=0.20 old-vs-clean validation comparison outcome

Implemented `scripts/analysis/delphi_k020_old_vs_clean_val_report.py`.

Command:

```bash
uv run --with scipy --with plotly --with pandas --with gcsfs \
  python scripts/analysis/delphi_k020_old_vs_clean_val_report.py
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report.html`
- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report_best_by_target.csv`
- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report_summary.csv`
- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report_1e22_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report_cached_old_val_check.csv`
- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report_fit.json`

Best comparable scale/LR fit by target, excluding anchor-calibration fits so targets are compared under the same
model family set:

| target | best model | heldout MAE | 1e22 MAE | 1e22 lr0.50 error | 1e22 lr0.67 error |
|---|---|---:|---:|---:|---:|
| old macro math val | per-LR log-log: `N` | 1.15% | 0.68% | +0.76% | +0.82% |
| new clean-seen loss | pooled Chinchilla + LR quadratic: `N + D_pre` | 1.44% | 2.11% | +2.47% | +1.33% |
| old eval/loss aggregate | pooled Chinchilla + LR quadratic: `D_math` | 6.49% | 11.73% | +12.66% | +10.45% |
| old 4plus math val | pooled Chinchilla + LR quadratic: `D_pre` | 7.05% | 12.94% | +13.91% | +11.64% |

The same-sweep old 4plus anchors match the older cached `math_val_loss` values closely where cells overlap:
about +0.01% to +0.03%, including +0.016% at 1e21/lr0.50 and +0.022% at 1e22/lr0.50. This makes a cache/key
mismatch unlikely. The result is more specific than "all old validation was bad": the old 4plus target and old
eval/loss aggregate extrapolate badly, while the macro aggregate and clean-seen target extrapolate well.

Verified with `./infra/pre-commit.py --changed-files --fix`.

Follow-up: reframed the HTML around the two primary targets only:

- old 4plus math val = `anchor_4plus_loss`, the earlier K=0.20 math validation target
- new clean-seen loss = `clean_seen_loss`

The `eval_loss` and `macro_loss` targets remain in collapsed auxiliary diagnostics because they are useful sanity
checks from the same eval jobs, but they are not the main old-vs-clean comparison. Regenerated and reopened
`sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report.html`; checks still pass.

Follow-up: removed the auxiliary diagnostics from the HTML entirely. The loss plot, target table, fit explorer,
and visible report tables now show only `anchor_4plus_loss` versus `clean_seen_loss`. The full diagnostic CSVs are
still written by the script, but the HTML no longer contains `eval_loss`, `macro_loss`, old eval/loss, or old macro
labels. Regenerated and reopened the report; checks still pass.

## 2026-06-21 - K=0.20 old-vs-clean validation comparison

Starting a comparison report for the K=0.20 p33m67 ladder that runs the same fit family against the old validation
losses and the new decontaminated clean-seen losses.

Inputs:

- New clean-seen CSV:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv`
- Old validation source: to be recovered from local analysis files/logbooks.

Planned artifact:

- `sk_midtrain_analysis_fable/delphi_k020_old_vs_clean_val_fit_report.html`

Follow-up: made report tables client-side sortable. The `All 1e22 endpoint predictions` table now includes both
signed error and absolute error, defaults to sorting by absolute error ascending, and every column header can be
clicked to sort numerically or lexicographically. Regenerated both combined and iso-token-only HTML/CSV artifacts.

## 2026-06-21 - K=0.20 clean-seen fit-family report

Starting a K=0.20-only fit-family report based on the new p33m67 clean-seen eval sweep:

- Input CSV:
  `gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv`
- Input rows: 36 = 9 scales x 4 LR factors.
- Target metric: `clean_seen_loss`.
- Holdout split: fit through `3e20`, score `1e21` and `1e22`.
- Primary reporting: explicit 1e22 prediction/error columns by LR.

Implemented `scripts/analysis/delphi_k020_clean_seen_fit_family_report.py`.

Command:

```bash
uv run --with scipy --with plotly --with pandas --with gcsfs \
  python scripts/analysis/delphi_k020_clean_seen_fit_family_report.py
```

Artifacts:

- `sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report.html`
- `sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_summary.csv`
- `sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_1e22_predictions.csv`
- `sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_fit_table.csv`
- `sk_midtrain_analysis_fable/delphi_k020_clean_seen_fit_family_report_fit.json`

Headline result: the best heldout model is the pooled Chinchilla fit with LR quadratic terms over `N + D_pre`.
It reaches 1.44% heldout MAE across the heldout `1e21` and `1e22` rows. On the explicit `1e22` rows, its
absolute errors are:

| LR factor | actual clean-seen loss | prediction | error |
|---|---:|---:|---:|
| 0.33 | 0.83148 | 0.86392 | +3.90% |
| 0.50 | 0.82499 | 0.84541 | +2.47% |
| 0.67 | 0.82340 | 0.83438 | +1.33% |
| 0.83 | 0.82477 | 0.83084 | +0.74% |

The per-LR Chinchilla fits over `C_pre` or `C_mid_math` are close by heldout MAE (1.70%) but are uniformly high
on the 1e22 rows by about 2.55% to 3.13%. The report includes sortable summary and 1e22 prediction tables,
plus dropdown-selectable Plotly views for every successful fit mode.

Verified with `./infra/pre-commit.py --changed-files --fix`.
