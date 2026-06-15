# Delphi true cooldown (cooldown20): Research Logbook

## Scope

- Goal: "true midtraining" for the Delphi ladder — resume each scale's pretrain
  at the **exact 80% step** (full optimizer state, original WSD LR schedule)
  and finish the last 20% of steps with **only the data mixture swapped** to the
  p33m67 math blend. Contrast with CPT (fresh warmup on the final model).
- Primary metrics: `eval/loss`, `eval/nemotron_cc_math_v1/macro_loss` on the
  byte-identical nemotron math val (contract in
  [`nemotron_math_data.md`](nemotron_math_data.md)).
- This file is the consolidated index for the cooldown thread. The raw
  operational history (per-poll babysitting, preemption recovery, every launch
  command) lives in [`midtraining_delphi.md`](midtraining_delphi.md); the
  original design + the May launch round in
  [`true_midtraining.md`](true_midtraining.md).

## Naming history (why two prefixes exist in W&B)

- **May round** (`true-midtrain-*`, project `marin`): first implementation per
  `true_midtraining.md`. Superseded — the 1e20 base was the wrong (v5) isoflop
  checkpoint (see the 🚨 banner in that file) and prefixes were not exact 80%
  steps.
- **June round** (`delphi-true-<base>-p33m67-cooldown20-a010`, project
  `delphi-midtraining`): relaunched on rewritten launcher
  (`experiments/midtrain_specs/true_midtrain/nemotron_math_only/launcher.py`)
  with exact-80% prefix checkpoints materialized by dedicated prefix runs.
  This is the canonical set.

## Two-stage pipeline

1. **Prefix runs** (`delphi-<base>-prefixes-qwen3*`, W&B project `marin`):
   re-train each scale's pretrain from an available pretrain checkpoint to the
   exact 80% step, to materialize a full-state (weights+optimizer) Qwen3
   checkpoint there. Permanent checkpoint is written only at the target step;
   intermediate autosaves go to `tmp/ttl=14d/checkpoints-temp/` (14-day TTL).
2. **Cooldown runs** (`delphi-true-<base>-p33m67-cooldown20-a010`, W&B project
   `delphi-midtraining`): resume from the prefix checkpoint, swap data to
   p33m67, run to the pretrain's natural end (last 20% of steps). Prefix
   checkpoint paths are wired through
   `experiments/midtrain_specs/true_midtrain/nemotron_math_only/configs/checkpoint_candidates.yaml`
   (launcher may only use human-reviewed rows).

## Results — cooldown20 a010 set (sealed 2026-06-07)

All 7 launched cells complete. Final eval metrics (W&B summaries, pulled
2026-06-11); math macro loss is strictly monotonic in scale.

| base | final step | eval/loss | math macro | paloma macro | W&B state | final checkpoint |
|---|---|---|---|---|---|---|
| 3e18 | 37,334 | 2.527 | 1.408 | 3.684 | finished | `gs://marin-us-east5/checkpoints/delphi-true-3e18-p33m67-cooldown20-a010/checkpoints/step-37334` + HF |
| 9e18 | 44,316 | 2.336 | 1.253 | 3.472 | finished | `…delphi-true-9e18…/step-44316` + HF |
| 2e19 | 55,124 | 2.241 | 1.178 | 3.364 | finished | `…delphi-true-2e19…/step-55124` + HF |
| 3e19 | 38,013 | 2.177 | 1.126 | 3.291 | finished | `…delphi-true-3e19…/step-38013` + HF |
| 9e19 | 40,282 | 2.038 | 1.012 | 3.135 | finished | `…delphi-true-9e19…/step-40282` + HF |
| 2e20 | 56,470 | 1.959 | 0.946 | 3.042 | crashed¹ | `…delphi-true-2e20…/step-56470` (no final HF²) |
| 3e20 | 35,509 | 1.909 | 0.904 | 2.985 | finished | `…delphi-true-3e20…/step-35509` + HF |

¹ 2e20 "crashed" is cosmetic: the job died at the very end (17 preemptions,
18 h runtime) after writing final tensors but before the top-level
`metadata.json` / W&B finish. Native `step-56470` passed
`assert_checkpoint_complete_for_model_type(..., model_type="qwen3", num_layers=21)`
(2026-06-04, re-verified since).
² 2e20's last HF export is `hf/step-50823`; the final-step HF export is an
open task if needed.

## Prefix checkpoint inventory (verified 2026-06-11)

All finished prefixes live in **permanent** storage and passed full Qwen3
structural validation (`assert_checkpoint_complete_for_model_type`) on
2026-06-11:

| base | layers | prefix checkpoint (exact 80%) | status |
|---|---|---|---|
| 3e18 | L11 | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-prefixes-qwen3/checkpoints/step-29868` | ✅ |
| 9e18 | L12 | `…/delphi-9e18-prefixes-qwen3/checkpoints/step-35453` | ✅ |
| 2e19 | L15 | `…/delphi-2e19-prefixes-qwen3/checkpoints/step-44100` | ✅ |
| 3e19 | L16 | `…/delphi-3e19-prefixes-qwen3/checkpoints/step-30411` | ✅ |
| 9e19 | L18 | `…/delphi-9e19-prefixes-qwen3/checkpoints/step-32226` | ✅ |
| 2e20 | L21 | `…/delphi-2e20-prefixes-qwen3-from40k/checkpoints/step-45113` | ✅ |
| 3e20 | L23 | `gs://marin-us-central2/checkpoints/delphi-prefix-checkpoints/delphi-3e20-prefixes-qwen3-v4c-r7-reserved32/checkpoints/step-28408` | ✅ |
| 1e21 | L26 | `gs://marin-us-central2/checkpoints/delphi-prefix-checkpoints/delphi-1e21-prefixes-qwen3-v4c-r11-reserved32-ram256/checkpoints/step-17645` | ✅ (closes the 2026-06-06 TODO) |
| 1e22 | L37 | `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-1e22-prefixes-qwen3-v5p32-east5-r17/checkpoints/step-30588` | ✅ SEALED 2026-06-12 (see r17 entries below) |

### 80%-exactness audit (2026-06-11)

Prefix step vs `trainer.num_train_steps` from each run's own config (cooldown
configs use the same totals). True 80% is fractional for most bases; prefixes
floored it.

| base | prefix step | total steps | actual % | delta from true 80% |
|---|---|---|---|---|
| 3e18 | 29,868 | 37,335 | 80.0000% | 0.0 steps |
| 9e18 | 35,453 | 44,317 | 79.9986% | −0.6 |
| 2e19 | 44,100 | 55,125 | 80.0000% | 0.0 |
| 3e19 | 30,411 | 38,014 | 79.9995% | −0.2 |
| 9e19 | 32,226 | 40,283 | 79.9990% | −0.4 |
| **2e20** | 45,113 | 56,477 | **79.8785%** | **−68.6** |
| 3e20 | 28,408 | 35,510 | 80.0000% | 0.0 |
| 1e21 | 17,645 | 22,057 | 79.9973% | −0.6 |
| 1e22 (r15 temp) | 30,392 | 38,235 | 79.4874% | −196.0 |

**2e20 anomaly:** its target (45,113) is exactly ⌊0.8 × 56,392⌋ — computed
from a different total (56,392, presumably the registry value when the YAML
was generated 2026-05-23) than the run configs' 56,477. Net effect on the
sealed set: 2e20's cooldown started 68.6 steps (−0.12%) early and ran 11,364
cooldown steps = 20.12% of pretrain. All other sealed cells are 80.00%
(−0.6 steps worst case, ≤0.003%).

## 1e22 prefix: salvage state (as of 2026-06-11)

Target: step **30,588** = exactly 80% of 38,235 pretrain steps (batch 1024 ×
seq 4096 ≈ 160.4B total pretrain tokens). No permanent checkpoint exists for
any 1e22 prefix attempt; surviving state:

| checkpoint | progress | written | expires (TTL) | notes |
|---|---|---|---|---|
| original pretrain `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/checkpoints/step-30000` | 78.46% | — | permanent | fallback floor; what r10/r15 bootstrapped from |
| r10 temp `…ttl=14d…/delphi-1e22-prefixes-qwen3-v4c-r10-reserved32-ram256/checkpoints/step-30154` | 78.86% | 2026-06-04 | **~2026-06-18** | committed (`metadata.json`) |
| r15 temp `…ttl=14d…/delphi-1e22-prefixes-qwen3-v4c-r15-reserved64-ram256-direct/checkpoints/step-30392` | 79.49% | 2026-06-05 | **~2026-06-19** | committed; passed Qwen3 L37 validation 2026-06-11; best resume floor (196 steps to target) |

**Resume-determinism evidence (2026-06-11):** r10 and r15 have byte-identical
`data`/`optimizer`/`model` config blocks (trainer.seed=0, feistel permutation,
same train/val split params) and their overlapping W&B histories replay the
same per-step train losses to ~1e-5 (step 30001: 2.495820 vs 2.495815) despite
different topologies (reserved32 vs reserved64). Levanter's step-indexed
deterministic loader ⇒ a resumed finisher continues the exact same data stream
and split. Acceptance gate for any finisher: resume from step-30392 and compare
logged losses on steps 30393–30412 against r15's W&B history (match to ~1e-4
⇒ on-trajectory; ~40 min of compute risked).

Hardware constraints on record: v4-32 pdev16 compile-OOMs (36.90G vs 30.75G
HBM); reserved capacity forbidden (2026-06-05 operator constraint). Proven
recipe for this model size (iso-token 1e22 CPT, 2026-06-11): preemptible
v5p-32 + `--per-device-parallelism 16` + 400g host RAM, ~115 s/step at batch
1024 ⇒ 196 steps ≈ 6.5 h (+ cross-region read of central2 caches if run on
v5p in east5/central1).

### 2026-06-11 ~17:16Z — 1e22 finisher launched (r17, us-east5-a v5p-32)

User greenlit finishing the prefix on us-east5-a v5p-32 with everything
in-region. Done this session:

1. **Staged r15 step-30392 to east5** (out of TTL — this is also the safety
   copy): `gsutil -m rsync` central2→east5, 83/83 objects, 108.57 GiB verified,
   then re-passed Qwen3 L37 structural validation on the copy. Path:
   `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-1e22-prefixes-qwen3-v4c-r15-reserved64-ram256-direct/checkpoints/step-30392`
2. **Staged launch metadata** next to it (`.executor_info` from the canonical
   1e22 root + `isoflop_analysis_result.json` from
   `adamh-scaling-ladder-nemotron-analysis-9200ec`) — required because the
   latest launcher guards (`08a7b1971`) refuse any cross-bucket read.
3. **Dry-run passed**: source 30,392 (79.49%) → target 30,588 (80.00%),
   196 steps, all paths east5, preemptible v5p-32, pdev16, ram 400g,
   5-min temp saves, region us-east5.
4. **Submitted coordinator** — first attempt
   (`…-20260612-001604`, mem 16GB) stuck pending: east5 CPU pool had only
   6.1GB free. Killed; resubmitted at 6GB as
   `/ahmedah/delphi-1e22-prefixes-qwen3-v5p32-east5-r17-from30392-20260612-002140`
   (cpu 0.5, mem 6GB, interactive, us-east5-a, no-preemptible coordinator).
   Coordinator ran preflight + submitted TPU child
   `…-002140/checkpoints-delphi-prefix-1e22-step30588-stop30589_ed1d00ab-efc771c2`
   at 00:22:17Z — child PENDING on v5p-32 coscheduling (4 workers).
   Capacity picture at submit time: the ONLY east5-a v5p-32 slice is held by
   `/tonyhlee/4b-nemotron-cascade2-…` (4B SFT, all 4 workers, running since
   Jun 8); scale group `tpu_v5p-preemptible_32-us-east5-a` has 40 consecutive
   autoscaler failures with a GCP zone stockout message. Child runs when the
   slice frees or capacity returns. Command:
   `uv run python scripts/materialize_delphi_prefix_checkpoint.py --base 1e22
   --source-step 30392 --target-step 30588 --source-checkpoint-path <east5
   staged> --no-source-mirror --source-executor-info-path <east5>
   --legacy-analysis-result-path <east5> --output-root
   gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-1e22-prefixes-qwen3-v5p32-east5-r17
   --tpu v5p-32 --ram 400g --per-device-parallelism 16 --region us-east5`
5. Final 80% checkpoint will land at
   `gs://marin-us-east5/.../delphi-1e22-prefixes-qwen3-v5p32-east5-r17/checkpoints/step-30588`.
   Expect the TPU child to queue while east5-a v5p is busy/tier-blocked
   (user accepted). ~196 steps ≈ 6.5 h at ~115 s/step once allocated.
   Acceptance gate: first ~20 logged train losses (steps 30393–30412) must
   match r15's W&B history to ~1e-4.

### 2026-06-12 ~02:30Z — r17 allocated, resumed, acceptance gate analyzed

- v5p-32 capacity freed after ~1 h queue; child RUNNING. All 4 workers logged
  `Loading checkpoint from gs://marin-us-east5/.../r15.../step-30392` and
  `Resuming training from step 30393`. Data caches read from
  `gs://marin-us-east5/tokenized/nemotron_cc/...` — fully in-region.
- Benign noise: one background W&B artifact-upload traceback
  (`FileNotFoundError .../config.yaml` in `levanter.tracker.background`) —
  tracker thread only, training unaffected. Cache loader printed a
  metadata-mismatch warning (`append_bos`, `max_length`) — stored cache
  metadata vs current defaults; superseded by the empirical gate below.
- **Loss-replay gate (calibrated): ON-TRAJECTORY.** W&B run
  `delphi-1e22-prefixes-qwen3-v5p32-east5-r17` (project `marin`) vs r15 over
  overlap steps 30393+. Step +1 matches to 1e-6 (first batch provably
  identical; batch-to-batch swing is ~1e-2). Divergence then grows smoothly/
  geometrically: mean|Δ| 4.4e-4 (offsets 1–5) → 4.2e-3 (6–15) → ~1.1e-2 (16).
  Known-good baseline r10-vs-r15 (both v4, certified same stream) shows the
  same geometric growth, slower: 1.2e-5 (1–5) → 1.8e-3/max 4.6e-3 (81–160).
  Faster growth here is expected from the chip-generation change (v4→v5p) +
  grad-accumulation order (pdev16). The naive scripted threshold
  (max|Δ|<1e-3 over 20 steps) tripped — it was mis-calibrated (even the v4/v4
  baseline violates it by offset ~40). No step in the discriminating window
  (offsets 1–8) exceeds the amplification envelope ⇒ same data stream, same
  schedule; residual difference is hardware-level numeric noise, the same
  class as any topology change/preemption-resume. **Decision: ride to 30588.**
- Implication for the sealed checkpoint: r17's step-30588 is the exact-80%
  prefix on the same data/schedule, bit-different from a hypothetical v4
  completion at ~1e-2 train-loss noise scale — inherent to running on v5p at
  all, accepted when the move to v5p-32 east5 was chosen.

### 2026-06-12 ~12:50Z — SEAL: 1e22 prefix DONE at exact 80%

- One mid-run zone-wide preemption (~02:54Z, GCP reclaimed all east5-a v5p-32;
  ~3 steps replayed from temp step-30430); resumed when capacity returned and
  ran to completion. Both child AND parent coordinator `succeeded` (no wrapper
  false-fail — the 3e20-era schema-checker patch held).
- Final checkpoint:
  `gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-1e22-prefixes-qwen3-v5p32-east5-r17/checkpoints/step-30588`
  — `metadata.json` committed 12:47:39Z (`is_temporary: false`, provenance
  records source step-30392 direct load). Independent
  `assert_checkpoint_complete_for_model_type(..., qwen3, num_layers=37)` PASSED.
- **All 9 ladder prefixes now exist at (floor-)exact 80%.** W&B:
  `delphi-1e22-prefixes-qwen3-v5p32-east5-r17` (project `marin`).

### 2026-06-12 ~16:00Z — full audit of checkpoint_candidates.yaml + row updates

Audited all 27 rows (user-requested deep check before 1e21/1e22 launch):
- **Paths**: every approved row's `suggested_checkpoint_path` exists with
  `metadata.json` (gsutil stat, 11/11 OK).
- **Arithmetic**: all rows consistent with `delphi_models.py` registry
  (targets, deltas, cooldown steps, path-step ↔ suggested_step, progress %)
  EXCEPT the two 2e20 rows: their targets were computed from
  `verified_checkpoint_step` 56,392 (where the ORIGINAL 2e20 pretrain actually
  stopped — 85 steps short of its 56,477 schedule) instead of the schedule.
  Root cause of the sealed −0.12% 2e20 start offset; 45,113 = ⌊0.8 × 56,392⌋.
- **NEW finding — 2e20 cooldown ended 7 steps short of schedule**: cooldown
  config `num_train_steps=56,477`, `stop_step=None`, keep-every-5,647; last
  logged step 56,469; `step-56470` is the 10th cadence save, crashed mid-commit
  (hence missing top-level metadata.json + missing final HF export), never
  resumed. Old logbook's "COMPLETE" overstates by 7 steps (0.06% of cooldown,
  ~29M tokens, last sliver of LR decay) — scientifically negligible, recorded
  for accuracy. Cheap to finish (resume 56470→56476) if exactness ever matters.
- Cosmetic: approved rows 9e19-cd20 and 2e20-cd20 lack the
  `materialized_checkpoint: true` field others have (launcher ignores it).
- Approved cooldown30 rows intentionally point at closest-available isoflop
  steps (not exact 70%) per the file header — fine as long as nobody assumes
  exactness.
- `p33m67.yaml` has planned cells for 1e21/1e22 cooldown20 keyed to the
  candidate IDs ✓.
- **Applied row edits** (mirroring the 3e20 Jun-4 precedent):
  `delphi-1e21-cooldown20` → r11 `step-17645` (exact_target, 79.9973%,
  approved); `delphi-1e22-cooldown20` → east5-r17 `step-30588` (exact_target,
  80.0000%, approved). **Launcher dry-runs PASS for both cells.**
- Launch gotcha for 1e21: its prefix is 37.81 GiB in us-central2; east5 launch
  needs `--allow-cross-region-stage --stage-budget-gb 45` (default budget is
  0). 1e22's prefix is already in east5 → in-region stage, no flags.

## Open items

1. ~~Copy 1e22 temps out of TTL~~ — DONE 2026-06-11 (east5 staging copy of
   step-30392 doubles as the safety copy; r10 step-30154 left in central2 temp,
   expires ~Jun 18, only needed if r15's copy is somehow bad).
2. ~~1e22 prefix finisher~~ — **DONE 2026-06-12**, sealed at step-30588
   (entry above).
3. **1e21 cooldown20 launch**: prefix is sealed+validated; the
   `delphi-1e21-cooldown20` row in `checkpoint_candidates.yaml` still points at
   old pretrain `step-20000` (+2355 off-target, `needs_human_review`) — update
   to `step-17645` (as done for 3e20 on 2026-06-04) before launch.
4. **1e22 cooldown20 launch**: blocked on (2); YAML row also needs updating to
   the finished prefix.
5. **2e20 final-step HF export** (`hf/step-56470`) if HF format is required.

## See also

- [`midtraining_delphi.md`](midtraining_delphi.md) — full operational log
  (launches, preemptions, resume rules, per-poll status) for the a010 set.
- [`true_midtraining.md`](true_midtraining.md) — design doc + May round
  post-mortem (wrong-1e20-base 🚨 rule).
- [`nemotron_math_data.md`](nemotron_math_data.md) — data/val contract,
  contamination analysis.
- [`debug_midtrain.md`](debug_midtrain.md) — val-loss crossover debug; CPT
  iso-token ladder (exp #7) sealed 2026-06-11 with no crossover.
- `.agents/projects/delphi_midtraining.md` — project-level plan/state.
