# bloomv2_m2: Data composition + generation methodology for M2 DPO LoRA

This doc consolidates everything that went into building the `bloomv2_m2`
preference dataset and the M2 DPO LoRA training run launched from it.

For the upstream discovery work (tension atlas → paired rubrics → BCG → M1
regression audit → 40-point seed selection → 10-point pilot → hand-audit →
Tier B scale-up), see `claude_stress_testing.md` Experiments 1–16.

This doc is the **data-centric view**: exactly what was generated, how
chosen/rejected were selected, and where the artifacts live.

---

## Run identity

- **Training run (M2)**: Iris job `/ahmed/m2-bloomv2-m2-20260421-090500`
  on `v5p-8` in `us-central1-a`, submitted 2026-04-21T09:05 UTC.
- **Branch**: `dpo-lora-clean`, local commit `5deecce0e`
  ("[dpo] M2: LoRA DPO on bloomv2_m2 (bloomv2 + pilot tension pairs), from SFT").
  *Not pushed to origin* — user reviews before push.
- **WandB project**: `marin-community/dpo`.
- **WandB run (expected URL — confirm in browser if needed)**:
  `https://wandb.ai/marin-community/dpo/runs/lora_m2_from_sft_bloomv2_m2_lr1e5_seed0_b64_v5p8-d33a24`
  (derived from the checkpoint directory slug; the controller's wandb
  init line did not surface in the Iris log tail, but the project + run
  name convention matches M1's pattern).
- **Checkpoints root**:
  `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_m2_from_sft_bloomv2_m2_lr1e5_seed0_b64_v5p8-d33a24/`
  - rolling at `checkpoints/step-<N>/`
  - HF exports at `hf/step-<N>/` every 200 steps.

---

## Dataset at a glance

**Path**: `gs://marin-us-central1/preference/bloomv2_m2/`

```
train/
  shard-00000..shard-00021.jsonl.gz   # bloomv2 base, 22 shards, unchanged
  shard-pilot.jsonl.gz                 # 2,898 tension pairs
val_deduped/
  shard-00000.jsonl.gz                # bloomv2 base val, unchanged
  shard-pilot.jsonl.gz                 # 325 tension pairs
README.md
```

**Strategy**: preserve the original bloomv2 shards bit-for-bit via
server-side `gcloud storage cp`; append a single new pilot shard to each
split. No re-tokenization or re-encoding of bloomv2 base.

**Record schema** (bloomv2-compatible):
```json
{
  "chosen":      [{"role": "user", ...}, {"role": "assistant", ...}],
  "rejected":    [{"role": "user", ...}, {"role": "assistant", ...}],
  "hash":        "<sha256[:24]>",
  "prompt":      "<user text>",
  "statement_id": "<pair_id>",    // e.g., "comply_with_laws__no_agenda"
  "question_id": "tension::<pair_id>::tp<NNN>::v<VI>::c<CI>::r<RI>"
}
```

### Size delta vs bloomv2

|                | bloomv2 base | bloomv2_m2  | delta          |
|----------------|-------------:|------------:|----------------|
| Train samples  | ~108,736     | ~111,634    | **+2,898 (+2.7%)** |
| Val samples    | ~small       | ~small+325  | +325           |
| Train shards   | 22           | 23          | +1 (shard-pilot) |
| Val shards     | 1            | 2           | +1 (shard-pilot) |

(Base count inferred from M1's 1699 training steps × batch=64 = 108,736
samples at 1 epoch on the original bloomv2 preference set.)

---

## End-to-end pipeline (what actually ran)

Every paid API call used the **OpenAI Batch API @ 50% off** with
`reasoning_effort="none"` on gpt-5.x.

### Phase 0 — seed selection (already done in Experiment 15)

- Start from 40-point M2 seed (post-Exp-15, with clause filter removed).
- Rubrics at `experiments/posttrain/stage3_output/paired_rubrics_seed_40.jsonl`
  (reconstructed from a cached OpenAI score batch; see Exp 14).

### Phase A — variants via gpt-4.1 (40 requests)

- Per point, ask gpt-4.1 for **10 paraphrase prompts** at the same
  `(peak_corner, tension_name)`.
- System prompt emphasises: preserve the trade-off, materially change
  surface wording, no softening, no safety rails.
- Batch ID: `batch_69e72c1bf0308190bf63684673a14981`.
- Cost: ~$0.50. Wall clock: ~3 min.
- **Output**: 400 variant prompts (40 points × 10 variants).
- Script: `experiments/posttrain/gen_train_variants.py submit/collect`.

### Phase B — gpt-5.1 chosen generation (2000 draws)

- For each variant: draw **5 responses** from gpt-5.1 with
  `reasoning_effort="none"`, temperature default, no rubric in the system
  prompt (the model does not know which statements apply — it just tries
  to be helpful).
- Batch ID: `batch_69e72d159dd88190acd2d4eb2181c91c`.
- Cost: ~$15. Wall clock: ~35 min.
- **Output**: 2000 candidate chosen responses.
- Script: `experiments/posttrain/gen_chosens.py gen-submit/gen-collect`.

### Phase C — M1 TPU inference on variants (4000 samples)

- Target model: `tune_lora_lr1e5_seed0_step1699` (M1 HF-reexported).
- For each variant: draw **10 M1 responses** at temperature 0.7.
- Iris v6e-4 preemptible, us-east5-b.
- Uploaded the 400 variant prompts to
  `gs://marin-us-east5/alignment/bloomv2_m2_variant_prompts/shard_00000.jsonl.gz`
  first via `build_variant_prompts_shard.py`.
- Iris job: `/ahmed/bloomv2-m2-m1-variants-20260421-075422`.
- Cost: ~$1. Wall clock: ~6 min (vLLM startup + generation).
- **Output**: 4000 M1-on-variants responses.

### Phase D — gpt-5.1 judges the 2000 chosens (4000 requests)

- Each chosen scored twice: once against `A_rubric`, once against
  `B_rubric`. Score 0–10 with strict JSON.
- Batch ID: `batch_69e735389c3c8190b604230ea8d45f39`.
- Cost: ~$12. Wall clock: ~28 min.
- Script: `experiments/posttrain/gen_chosens.py judge-submit/judge-collect`.

### Phase E — gpt-5.1 judges the 4000 M1 variants (8000 requests)

- Each M1 response scored twice against the same paired rubric.
- Uses a variant-aware custom-id (`m1var_score::…::v<VI>::s<SI>::<side>`)
  so the K=10-per-variant samples don't collide across variants.
- Batch ID: `batch_69e72eea5c748190a48eb909769eb061`.
- Cost: ~$20. Wall clock: ~39 min.
- Script: `experiments/posttrain/judge_m1_variants.py submit/collect`.

### Phase F — select + assemble (local)

- **Chosen selection** (per variant): top-3 draws conditional on
  `min(A,B) ≥ 7`, sorted by `(-min(A,B), -(A+B), draw_idx)`.
  Yield: **907 chosens** across 329 viable variants (82% of variants
  produced ≥ 1 qualifying chosen).
- **Rejected selection** (per variant): from 10 M1 samples, keep only
  those with `joint_satisfied == False AND failed_side ≤ 5`, then apply
  **failure-mode clustering** (A-only / B-only / both) and round-robin
  across modes up to K=5. Yield: **1,739 rejecteds** across 370 variants
  (92.5% had ≥ 1 qualifying rejected).
- **Crossproduct per variant**: up to 3 chosens × 5 rejecteds = 15 pairs.
  Yield histogram: 214 variants with full 15, 85 with partial yield,
  101 with 0 (those variants had no qualifying chosen or no qualifying
  rejected — see mechanics discussion below).
- **Raw pair count**: **3,793 preference pairs**.
- Scripts:
  `experiments/posttrain/gen_chosens.py select --top-k 3`,
  `experiments/posttrain/build_rejecteds_from_variants.py`,
  `experiments/posttrain/assemble_tier_b_pairs.py`.

### Phase G + H — merge with bloomv2 + GCS upload

- Train/val split: `variant_idx == 0` → val; `variant_idx ∈ {1..9}` →
  train.
- Dedup by hash within each split. After dedup:
  - **2,898 train preference pairs**
  - **325 val preference pairs**
- bloomv2 base shards copied server-side (no re-encode).
- Uploaded to `gs://marin-us-central1/preference/bloomv2_m2/`.
- Script: `experiments/posttrain/merge_with_bloomv2.py --upload`.

### Phase I — M2 training submission (subagent)

- Subagent checked out `dpo-lora-clean`, copied two staged drafts, ran
  `py_compile`, committed locally, submitted Iris job.
- First attempt failed (cross-region: `v5p-8` landed on `us-east5-a` but
  bloomv2_m2 lives in `us-central1`). Manual retry with `--zone us-central1-a`.
- Branch commit: `5deecce0e` (not pushed).
- Iris job: `/ahmed/m2-bloomv2-m2-20260421-090500`.
- Config: **verbatim M1** from `dpo-lora-clean` `experiments/tune_lora/common.py`
  (see next section).

### Total data-gen cost

~$50 all-in on OpenAI batch + ~$4 on TPU preemptible.

---

## M2 training config (verbatim M1)

```python
SimpleDPOConfig(
    resources=ResourceConfig.with_tpu("v5p-8", ram="400g"),
    per_device_eval_parallelism=DPO_EVAL_PARALLELISM["v5p-8"],  # 16
    train_batch_size=64,
    num_epochs=1.0,
    learning_rate=1e-5,
    lr_schedule="cosine",
    warmup=0.1,
    wandb_project="dpo",
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-instruct",   # SFT base
    adapter=LoraAdaptationConfig(
        r=64, alpha=64, dropout=0.0, zero_init_b=True, target_modules=None,
    ),
    reference=AdapterBaseReferenceConfig(),                    # ref = SFT
    train_seq_len=4096,
    max_seq_len=4096,
    beta=0.1,
    validation_split_fraction=None,
    reference_eval_cache=ReferenceEvalCacheConfig(mode="build_or_load"),
    steps_per_checkpoint=200,
    steps_per_hf_export=200,
    hf_generation_eos_token_ids=LLAMA3_CHAT_STOP_TOKEN_IDS,
    seed=0,
)
```

Only difference vs M1's run: the dataset path. Everything else
byte-for-byte identical to how M1 was trained, so any eval delta vs M1
is attributable to the bloomv2_m2 data mixture, not to config drift.

Training target: **1 epoch over bloomv2_m2** = 1,750 optimization steps.

---

## How chosen + rejected were picked (the contract)

This section captures the selection policy so anyone reading downstream
can see exactly what the DPO loss is pushing toward and against.

### Chosen

Per variant, from 5 gpt-5.1 draws:

1. Judge each draw twice — once against `A_rubric`, once against
   `B_rubric`. Two scores per draw: A ∈ [0,10], B ∈ [0,10].
2. Filter to draws with `min(A, B) ≥ 7` — **requires both sides to
   clear threshold**.
3. Sort by `(-min(A,B), -(A+B), draw_idx)` and take **top-3**.

**Property**: chosen selection explicitly rewards joint satisfaction.
A draw with A=9, B=9 beats A=10, B=8 beats A=10, B=7. The `min`
primary key means we never pick a response that aces one rubric while
middling on the other.

### Rejected

Per variant, from M1's 10 samples on that exact variant prompt:

1. Tag each sample's failure mode based on which rubric(s) fell below 7:
   - `A` — A<7, B≥7 (sacrificed A for B)
   - `B` — B<7, A≥7 (sacrificed B for A)
   - `both` — A<7 AND B<7 (failed both; not a trade-off, just bad)
2. Filter to `joint_satisfied == False AND min_side_score ≤ 5` (the
   **margin rule** — must be clearly below chosen, not borderline).
3. **Failure-mode clustering**: if multiple modes are present for this
   variant, round-robin across modes until K=5 reached. If only one
   mode present, take bottom-5 by `min_side_score`.

**Property**: rejected selection does not favor one side over the other.
Whenever M1 exhibits both A-biased and B-biased failures on the same
variant, we sample from both modes.

### Does the training set favor one trade-off direction?

**No, by design.** Two layered mechanisms ensure symmetry:

1. Chosen requires `min(A, B) ≥ 7` — you cannot win by acing A and
   bombing B, or vice versa.
2. Rejected round-robins across failure modes — A-fails and B-fails
   both appear in training, so DPO pushes the policy away from
   *either* direction of trade-off.

DPO loss per pair:
```
L = -log σ( β · (logπ(chosen) - logπ(rejected)) )
```
pushes the policy **up on jointly-satisfying responses** and **down on
whichever failure mode the rejected exhibits**. Since rejecteds include
both A-fails and B-fails, the resulting gradient penalises *any*
trade-off, not a preferred direction.

### Subtlety worth naming

The round-robin deliberately **over-represents minority failure modes**
relative to their natural frequency. If M1's character drift makes 9 of
10 samples fail on side A (over-refuse) and 1 of 10 fails on side B,
my selection takes one from each cluster — so DPO sees A-fail and
B-fail rejecteds at ~1:1 instead of the natural 9:1.

- **Good**: prevents M2 from simply swapping failure modes
  ("solve over-refusal by becoming under-refusing"). Trains against
  the full geometry of failures.
- **Risk**: dilutes the signal on M1's actual dominant failure mode.
  If M1 is overwhelmingly over-refusing and we only show DPO half as
  many over-refuse rejecteds as we could, we may learn less
  aggressively against the actual drift.

Experiment 13's regression analysis showed DPO (M1) systematically
sacrifices `no_agenda` / `letter_and_spirit` / `support_programmatic_use`,
so in practice most rejecteds should land in the same failure cluster.
The round-robin matters only for the minority of variants where M1
exhibits mixed failure modes.

**If M2 ships residual over-refusal on the 40 atlas points**, that is
the signal to drop the round-robin and go pure bottom-K-by-`min_side`
on the dominant failure mode.

---

## Known caveats

1. **Signal-to-noise ratio.** Pilot is 2,898 train pairs / ~111,634 total
   train pairs = 2.7%. At batch=64 over 1 epoch, each pilot pair is seen
   exactly once. If the tension signal isn't strong enough, M2 ≈ M1 on
   the 40-atlas eval. First ablation would be oversampled pilot
   (replicate `shard-pilot` 10× for ~22% pilot share).

2. **Reference = SFT** (`AdapterBaseReferenceConfig` over
   marin-8b-instruct). DPO loss measures divergence from SFT, not from
   M1. So M2 is trained fresh on bloomv2 + pilot, *not* continuing from
   M1. Per user directive. This means we are not preserving M1's
   specific wins by construction; the new data's marginal influence
   determines whether we improve.

3. **Judge consistency on `no_agenda` points** — flagged in Experiment 15
   as a risk (rubric may be noisier on this meta-ish clause) but not
   specifically measured for Tier B. Worth spot-checking post-training
   if M2 shows high variance on `no_agenda` corners in the eval.

4. **Original atlas prompts are frozen for evaluation.** Training used
   10 gpt-4.1-generated variants per point at the same peak_corner; the
   40 original atlas prompts are *not* in `bloomv2_m2` and remain the
   clean held-out eval set.

5. **Subagent used a `--zone us-central1-a` retry.** First submission
   hit a cross-region executor error (`v5p-8` landed in `us-east5-a`
   but dataset in `us-central1`). Not a config issue — just Iris
   scheduler zone selection. Retry succeeded immediately.

---

## Reproducibility pointers

All scripts live in `experiments/posttrain/` on the `alignment_function`
branch (the data-gen branch). Training scripts live on `dpo-lora-clean`.

Run, in order:
1. `gen_train_variants.py submit --pilot-index <40-pt-index> --n-variants 10`
2. `gen_train_variants.py collect`
3. `gen_chosens.py gen-submit --k-draws 5`
4. `gen_chosens.py gen-collect`
5. `build_variant_prompts_shard.py` + upload to GCS + launch `bcg_probe_infer.py`
   on TPU with `--n-samples 10`
6. `gen_chosens.py judge-submit` + `judge-collect` + `select --top-k 3`
7. `judge_m1_variants.py submit` + `collect`
8. `build_rejecteds_from_variants.py`
9. `assemble_tier_b_pairs.py`
10. `merge_with_bloomv2.py --upload`
11. (subagent on `dpo-lora-clean`) copy drafts from
    `.agents/projects/m2_submission_drafts/` + submit Iris job

---

## Artifacts

- `experiments/posttrain/stage4_output/tier_b/variants/train_variants_tier_b.jsonl`
- `experiments/posttrain/stage4_output/tier_b/chosens/chosens_gen.jsonl`
- `experiments/posttrain/stage4_output/tier_b/chosens/chosens_scores.jsonl`
- `experiments/posttrain/stage4_output/tier_b/chosens_tier_b.jsonl` (selected chosens)
- `experiments/posttrain/stage4_output/tier_b/m1_variants/generations.jsonl`
- `experiments/posttrain/stage4_output/tier_b/m1_variants/scores.jsonl`
- `experiments/posttrain/stage4_output/tier_b/rejecteds_tier_b.jsonl`
- `experiments/posttrain/stage4_output/tier_b/pairs_tier_b.jsonl` (3,793 raw pairs)
- `experiments/posttrain/stage4_output/tier_b/dataset/bloomv2_m2/` (local merged dataset)
- `gs://marin-us-central1/preference/bloomv2_m2/` (uploaded)
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_m2_from_sft_bloomv2_m2_lr1e5_seed0_b64_v5p8-d33a24/` (training output)
- `/tmp/n10_gate/state/SUMMARY.md` (live run summary)
- `/tmp/n10_gate/state/heartbeat.txt` (last-activity UTC timestamp)

## Next steps

1. Let M2 finish (ETA ~23:05 UTC). Final HF export at `hf/step-1750/`.
2. BCG-style eval on the 40 original atlas prompts at N=10 for M2.
3. Compare per-point vs M1: JSR, marginal A/B scores, character-drift-
   clause-specific deltas.
4. If signal is weak (M2 ≈ M1), repeat with oversampled pilot.
5. If signal is strong (M2 recovers the sacrificed clauses without
   losing M1's wins), write paper section and close the loop.
