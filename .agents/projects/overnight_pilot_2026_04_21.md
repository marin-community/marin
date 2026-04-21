# Overnight plan: Tier B data + bloomv2_m2 + M2 DPO LoRA — HARDENED (2026-04-21)

**Goal:** M2 finished training by morning. Data at Tier B (6000 pair ceiling).

**Robustness strategy:**
- All scripts written, compile-checked, AND dry-run BEFORE any batch fires.
- Each phase writes a success marker file; idempotent — re-runnable safely.
- Explicit error files on failure; wakeup reads markers + errors, advances.
- Parallel phases coordinated by independent state; no phase blocks on unrelated failure.
- Heartbeat file updated every wakeup so user can see last-advance time.
- Subagent has retry logic + clear error reporting.
- Pre-flight health checks before each expensive step.

---

## State directory: `/tmp/n10_gate/state/`

```
state/
  heartbeat.txt                  # updated every wakeup with UTC timestamp + last phase advanced
  phase_A_variants.done          # written after variant batch collected
  phase_A_variants.batch_id      # written at submit
  phase_A_variants.error         # written on any error
  phase_B_chosens_gen.done       # chosen gen collected
  phase_B_chosens_gen.batch_id
  phase_C_m1_tpu.done            # M1 on variants complete
  phase_C_m1_tpu.job_id          # Iris job ID
  phase_C_m1_tpu.error
  phase_D_chosens_judge.done
  phase_D_chosens_judge.batch_id
  phase_E_m1_judge.done
  phase_E_m1_judge.batch_id
  phase_F_assembled.done         # pairs file written
  phase_G_merged.done            # bloomv2 + pilot merged, written locally
  phase_H_uploaded.done          # uploaded to GCS as bloomv2_m2
  phase_I_m2_submitted.done      # subagent done, training job running
  phase_I_m2_submitted.job_id
```

Each phase's driver:
1. Check if `.done` already exists → skip, return immediately.
2. Check if prereqs exist → if not, return (wait for wakeup).
3. Run the operation. On success, write `.done`. On failure, write `.error` with details.

---

## Phase 0 — Pre-flight (do before any batch fires)

### 0.1 Write all Tier B scripts
- Modify `gen_train_variants.py`: add `--n-variants N` + `--skip-existing path.jsonl` to reuse 30 pilot variants.
- Modify `gen_chosens.py`: add `--select-top-k K` + accept variant file with >1 variant per (pair, tp).
- New script `build_variant_prompts_shard.py`: convert variants → GCS-shard for TPU.
- New script `build_rejecteds_from_variants.py`: select rejecteds per variant from M1-on-variants.
- Modify `assemble_pilot_dpo.py`: support crossproduct (top-K chosens × top-M rejecteds per variant).
- Modify `merge_with_bloomv2.py`: accept multiple shards; emit sharded output for large train splits.

All scripts compile-check: `python3 -m py_compile experiments/posttrain/*.py`.

### 0.2 Dry-run sanity checks (no API calls)
- Unit-check `gen_train_variants.py --skip-existing`: should skip 30 pilot variants, propose 370 new.
- Unit-check `build_variant_prompts_shard.py` on 5 variants → shard → inspect first record schema.
- Unit-check `merge_with_bloomv2.py --dry-run` with a synthetic 10-pair input.

### 0.3 Health checks (prove we can actually do each op)
- OpenAI: `client.files.list(limit=1)` — verifies API key.
- GCS read: list one shard of existing bloomv2 train.
- GCS write: `gcloud storage cp /tmp/ping gs://marin-us-east5/alignment/.ping` then delete.
- Iris: `iris job list | head -5` — verifies controller reachable.
- TPU availability: check recent v6e-4 jobs succeeded in us-east5-b.

### 0.4 Build 40-point index
- Already done: `/tmp/n10_gate/tier_b_index_40pts.json`.

### 0.5 Write `driver.sh` — single-entry overnight orchestrator
- Idempotent: reads state/, advances whichever phase is ready.
- Called by each ScheduleWakeup.
- Phases B+C run in parallel (both require A done but don't block each other).
- Phases D+E run in parallel (D requires B, E requires C; don't block each other).
- Phases F–I run serially after D+E.

---

## Phase A — Variants (gpt-4.1 batch)

- **Requests:** 370 new (40 points × 10 − 30 existing).
- **Reuse:** load 30 existing pilot variants from `pilot_dpo/train_variants_pilot_10pt.jsonl`.
- **Script:** `gen_train_variants.py submit --n-variants 10 --pilot-index tier_b_index_40pts.json --skip-existing pilot_dpo/train_variants_pilot_10pt.jsonl --job-dir tier_b/variants`
- **Estimated wall:** 10–20 min.
- **Success marker:** `state/phase_A_variants.done` (after collect).
- **Failure mode:** if <80% of 370 return clean, flag error. Fallback: proceed with what we got.

## Phase B — Chosens gen (gpt-5.1 batch, parallel with C)

- **Requests:** 1850 new (400 × 5 − 150 existing).
- **Reuse:** load 150 existing pilot chosens from `pilot_dpo/chosens/chosens_gen.jsonl`.
- **Script:** `gen_chosens.py gen-submit --variants tier_b/variants/train_variants_tier_b.jsonl --job-dir tier_b/chosens_gen --skip-existing pilot_dpo/chosens/chosens_gen.jsonl --k-draws 5`
- **Estimated wall:** 30–90 min (gpt-5.1 reasoning_effort=none, 2000 gens total).
- **Success marker:** `phase_B_chosens_gen.done` (after collect — merge old+new).

## Phase C — M1 TPU on variants (Iris job, parallel with B)

- **Preprocess:** `build_variant_prompts_shard.py` → local shard → upload to
  `gs://marin-us-east5/alignment/bloomv2_m2_variant_prompts/shard_00000.jsonl.gz`.
- **Submit iris job:**
  ```bash
  uv run iris --config lib/iris/examples/marin.yaml job run \
    --tpu v6e-4 --zone us-east5-b \
    --cpu 32 --memory 32GB --disk 100GB \
    --job-name bloomv2-m2-m1-variants-<ts> \
    --no-wait -- \
    uv run python experiments/posttrain/bcg_probe_infer.py \
      --target tune_lora_lr1e5_seed0_step1699 \
      --region us-east5 --tpu-type v6e-4 \
      --prompts-relative-path alignment/bloomv2_m2_variant_prompts \
      --step-suffix _bloomv2_m2_variants_n10 \
      --n-samples 10
  ```
- **Estimated wall:** 15–45 min (cold start + 400 prompts × 10 samples).
- **Pull result:** `gs://marin-us-east5/eval/bcg_probe_tune_lora_lr1e5_seed0_step1699_useast5_v6e4_bloomv2_m2_variants_n10/inference-<hash>/shard_00000.jsonl.gz`.
- **Success marker:** `phase_C_m1_tpu.done`.
- **Failure mode:** TPU preempted → Iris retries automatically. If 3 retries fail,
  flag error. Fallback: use cached M1-on-original-prompts (prompt-mismatched but
  same corner; 400 rejecteds available).

## Phase D — Judge 2000 chosens (gpt-5.1 batch, parallel with E)

- **Requests:** 2000 × 2 rubrics = 4000. Reuses existing 300 judge records for the 150 pilot chosens.
- **Script:** `gen_chosens.py judge-submit --job-dir tier_b/chosens_gen --skip-existing pilot_dpo/chosens/chosens_scores.jsonl`
- **Estimated wall:** 20–60 min.
- **Success marker:** `phase_D_chosens_judge.done`.

## Phase E — Judge 4000 M1-on-variants (gpt-5.1 batch, parallel with D)

- **Requests:** 4000 × 2 = 8000.
- **Script:** reuse `stage4_bcg_eval.py score-submit` pointing at:
  - `--rubrics experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl`
  - `--job-root experiments/posttrain/stage4_output/bloomv2_m2_m1_variants/`
- **Estimated wall:** 30–90 min.
- **Success marker:** `phase_E_m1_judge.done`.

## Phase F — Select + assemble (local, fast)

1. Select top-3 chosens per variant (from 5 draws, conditional on `min(A,B) ≥ 7`).
2. Select bottom-5 rejecteds per variant (from 10 M1 samples, `joint_satisfied=False AND failed_side ≤ 5`, failure-mode clustering).
3. Crossproduct per variant: up to 3 × 5 = 15 pairs.
4. Expected yield: 3000–6000 pairs after drops.
- **Script:** `assemble_pilot_dpo.py` (extended to take tier_b chosen/rejected files).
- **Output:** `tier_b/pairs_tier_b.jsonl`.
- **Success marker:** `phase_F_assembled.done`.

## Phase G — Merge with bloomv2 (local, medium)

1. Download bloomv2 base train/val shards (11 train shards + 1 val shard).
2. Convert tier_b pairs to bloomv2 record schema.
3. Train/val split: variant_idx=0 → val, rest → train.
4. Dedup by hash within split.
5. Concatenate base + tier_b. Reshard train to <50k records/shard (bloomv2 has ~500k total).
- **Script:** `merge_with_bloomv2.py` (extended for multi-shard output).
- **Output:** `stage4_output/pilot_dpo/dataset/bloomv2_m2/{train,val_deduped}/`.
- **Success marker:** `phase_G_merged.done`.

## Phase H — Upload to GCS (remote, ~1 min)

- `gcloud storage cp -r stage4_output/pilot_dpo/dataset/bloomv2_m2/* gs://marin-us-central1/preference/bloomv2_m2/`
- **Verify** after: `gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/{train,val_deduped}/ | head`.
- **Success marker:** `phase_H_uploaded.done`.

## Phase I — M2 training submission (subagent)

See "Subagent prompt" below. Subagent checks out `dpo-lora-clean`, copies 2 draft files, commits locally (no push), submits iris job.

- **Iris job ID:** written to `phase_I_m2_submitted.job_id`.
- **On failure:** write `phase_I_m2_submitted.error` with details. Next wakeup re-tries subagent once.
- **Max 2 subagent attempts.** If both fail, halt, heartbeat file will be stale, user sees.

---

## Subagent prompt (Phase I) — exact text

```
You are a deployment subagent. The user has pre-authorized this action.

Goal: on the dpo-lora-clean branch, copy two files into place and submit an
Iris TPU LoRA DPO training job. Return iris job ID + initial status.

Target branch: dpo-lora-clean (M1 was trained on this branch; same code base).
Target dataset: gs://marin-us-central1/preference/bloomv2_m2/

Steps (if any step fails, write /tmp/n10_gate/state/phase_I_m2_submitted.error
with full detail and HALT — do not guess or continue):

1. Verify dataset is live:
     gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/train/ | head
     gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/val_deduped/ | head
   Both must show at least one shard-*.jsonl.gz.

2. Schema spot-check: download one val record, verify schema has
   {chosen, rejected, hash, prompt, statement_id, question_id} and chosen is
   [user-turn, assistant-turn].

3. Check out the branch (this is on the main repo, not the worktree):
     cd /Users/ahmed/code/marin
     git fetch origin dpo-lora-clean
     git checkout dpo-lora-clean
     git pull origin dpo-lora-clean
   If the working tree is dirty, HALT with the list of dirty files.

4. Copy drafts into place:
     DRAFT_DIR=/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/.agents/projects/m2_submission_drafts
     cp $DRAFT_DIR/dpo_bloomv2_m2.py experiments/
     cp $DRAFT_DIR/m2_from_sft_beta0p1_lr1e5.py experiments/tune_lora/

5. Compile check:
     python3 -m py_compile experiments/dpo_bloomv2_m2.py experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py

6. Commit (DO NOT push):
     git add experiments/dpo_bloomv2_m2.py experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py
     git commit -m "[dpo] M2: LoRA DPO on bloomv2_m2 (bloomv2 + pilot tension pairs), from SFT"

7. Submit iris job with EXACT M1 training config (the drafts use M1's config
   verbatim — see experiments/tune_lora/common.py for the canonical reference):
     uv run iris --config lib/iris/examples/marin.yaml job run \
       --tpu v5p-8 --zone us-east5-a \
       --cpu 32 --memory 400GB --disk 100GB \
       --job-name m2-bloomv2-m2-$(date -u +%Y%m%d-%H%M%S) \
       --no-wait -- \
       uv run python experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py

   Capture the job ID from stdout.

8. Verify: sleep 90s, run `uv run iris --config lib/iris/examples/marin.yaml job list | grep <id>`.
   If status is FAILED in <2 min, fetch bug-report, include in response.

9. Write /tmp/n10_gate/state/phase_I_m2_submitted.done with the job ID and
   submit time (UTC).

Return to the main conversation: job ID, submit time, initial status,
any warnings encountered.

Constraints: DO NOT push to origin. DO NOT modify any file outside of
experiments/. DO NOT touch the original bloomv2 dataset path. DO NOT rerun
if phase_I_m2_submitted.done already exists — just return its contents.
```

---

## Robustness mechanisms

### R1: Idempotency
Every phase checks `state/phase_X_*.done` before doing work. Running a wakeup twice is harmless.

### R2: Heartbeat
Every wakeup writes `state/heartbeat.txt` with UTC timestamp + "advanced phase X" or "waiting on phase Y". If user wakes and the timestamp is > 2h old, something's wrong.

### R3: Parallel phase coordination
Wakeups check ALL pending phases (A done → start B+C parallel; both B+C done → start D+E parallel; both D+E done → F→G→H→I serial). No single phase blocks unrelated work.

### R4: Bounded retries
- Batch polling: up to 12 wakeups per batch (~4–6h max). If still not done, flag error.
- Iris TPU: automatic retry via Iris scheduler. Up to 3 attempts.
- Subagent: 2 attempts total. After that, halt.

### R5: Fallback for M1 TPU (Phase C)
If TPU job fails 3× or is preempted for >2h:
- Fall back to using cached M1-N=10-on-original-prompts (400 samples).
- This loses prompt-match but keeps Tier B data volume at ~60% (original prompts cover 40 points × 10 samples, not 400 variants).
- Document as known caveat in README.

### R6: Pre-flight health checks
Before kicking off Phase A, verify:
- OpenAI API reachable (`client.files.list(limit=1)`).
- GCS write access (ping file to marin-us-east5).
- Iris controller reachable (`iris job list | head`).
- TPU pool has capacity (no recent v6e-4 preemption storm in us-east5-b).

If any fails, write `state/preflight.error` and halt. User debugs in morning.

### R7: Error visibility
All errors go to `state/phase_X.error`. Final summary writes `state/SUMMARY.md` with everything that ran, timings, any errors, final pair count, iris job ID.

---

## Wakeup cadence

- T+0 min: pre-flight + Phase A submit.
- T+15 min: poll A; submit B + C in parallel if A done.
- T+35 min: poll B + C; advance judge D or E if parent ready.
- T+65 min: continue polling/advancing.
- T+90 min: most expected-case Tier B data is ready.
- T+100 min: Phase F+G+H in rapid sequence.
- T+110 min: Phase I subagent.
- T+120 min onwards: M2 training running. Subsequent wakeups just update heartbeat + check training job status.

---

## What user wakes up to (happy path)

- `state/SUMMARY.md`: final numbers, pair count, timings, GCS path, iris job ID.
- `state/heartbeat.txt`: last update within 30 min of wakeup.
- `gs://marin-us-central1/preference/bloomv2_m2/`: final dataset.
- M2 iris job either still running (on v5p-8, 3–5h total from submit) OR done with HF-exported checkpoint at `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_m2_from_sft_bloomv2_m2_beta0p1_lr1e5_seed0_b64_v5p8-<hash>/`.
- Logbook Experiment 16 appended.
- Two files committed locally on `dpo-lora-clean` (not pushed — user reviews and pushes).

## What user wakes up to (sad path)

- `state/heartbeat.txt` stale (>2h).
- `state/phase_X.error` with the specific failure.
- `state/SUMMARY.md` with what got done, what didn't, and a one-line "here's how to manually resume" per failed phase.

---

## Budget (actual, tracked to `state/SUMMARY.md`)

- Phase A (gpt-4.1 variants): ~$2
- Phase B (gpt-5.1 chosens): ~$15
- Phase C (v6e-4 TPU, preemptible): ~$3
- Phase D (gpt-5.1 judge chosens): ~$10
- Phase E (gpt-5.1 judge M1 variants): ~$20
- Phase I (v5p-8 LoRA DPO training): ~$50 for 5h
- **Grand total: ~$100.**

---

## Order of operations tonight (next ~45 min before sleep)

1. Write all Tier B script modifications (~25 min).
2. Compile-check all scripts.
3. Pre-flight health check.
4. Write `driver.sh` with state-machine.
5. Submit Phase A (gpt-4.1 variants).
6. Set first wakeup for T+15 min.
7. Commit plan + scripts to a snapshot for safety.
8. User goes to bed.
