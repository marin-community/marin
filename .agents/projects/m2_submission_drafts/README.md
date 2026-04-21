# M2 submission drafts (from-SFT, bloomv2_m2)

Two files staged for `dpo-lora-clean`. An overnight subagent will copy them
onto that branch, commit, and submit the iris job.

## Directives (from user, 2026-04-21)

1. **Keep old bloomv2 dataset unchanged.** Do not touch
   `gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/`.
2. **Create a new dataset named `bloomv2_m2`** that combines bloomv2 base +
   the 10-point pilot tension preference pairs.
3. **Train from SFT** (marin-8b-instruct), NOT from M1. Same config as M1,
   swap the dataset only.
4. **User-authorized:** subagent may check out dpo-lora-clean and submit the
   training job once the dataset is uploaded to GCS.

## Files and target paths

| draft | target on dpo-lora-clean |
|---|---|
| `dpo_bloomv2_m2.py` | `experiments/dpo_bloomv2_m2.py` |
| `m2_from_sft_beta0p1_lr1e5.py` | `experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py` |

## Deployment sequence (subagent)

```bash
git fetch origin
git checkout dpo-lora-clean
git pull origin dpo-lora-clean

# Copy drafts into place.
cp .agents/projects/m2_submission_drafts/dpo_bloomv2_m2.py experiments/
cp .agents/projects/m2_submission_drafts/m2_from_sft_beta0p1_lr1e5.py experiments/tune_lora/

# Sanity: compile both.
python3 -m py_compile experiments/dpo_bloomv2_m2.py experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py

# Commit.
git add experiments/dpo_bloomv2_m2.py experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py
git commit -m "[dpo] M2: LoRA DPO on bloomv2_m2 (bloomv2 + 10-pt pilot), from SFT"
```

## Pre-submit sanity checks

```bash
# Dataset is live on GCS.
gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/
gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/train/
gcloud storage ls gs://marin-us-central1/preference/bloomv2_m2/val_deduped/

# Record schema intact.
gcloud storage cp gs://marin-us-central1/preference/bloomv2_m2/train/shard-00000.jsonl.gz /tmp/shard-check.jsonl.gz
gzip -dc /tmp/shard-check.jsonl.gz | head -1 | python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); assert set(r) >= {"chosen","rejected","hash","prompt","statement_id","question_id"}; print("schema OK")'

# SFT base model reachable (should not need to download, just verify.)
gcloud storage ls gs://marin-us-central1/checkpoints/marin-community--marin-8b-instruct/ 2>&1 | head || true
```

## Submit command

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --tpu v5p-8 \
  --zone us-east5-a \
  --cpu 32 \
  --memory 400GB \
  --disk 100GB \
  --job-name m2-from-sft-$(date -u +%Y%m%d-%H%M%S) \
  --no-wait \
  -- \
  uv run python experiments/tune_lora/m2_from_sft_beta0p1_lr1e5.py
```

## Budget

- v5p-8 preemptible ≈ $8-12/hour depending on region.
- M1 original run: ~4-6 hours for 1 epoch at batch=64.
- bloomv2_m2 is bloomv2 + ~60 pairs = effectively same size.
- Expected: **~5h TPU ≈ $50-60**.

## Caveats (worth rereading in the morning)

### 1. Signal-to-noise on the pilot pairs

60 pilot pairs in ~500k total = 0.01% of training data. At 1 epoch with
batch=64, the model sees each pilot pair exactly once, buried in the stream.
The pilot signal may be **too weak to measurably shift M2 vs M1**.

This is a deliberate conservative first baseline per user directive —
"same config as M1, swap dataset only." If M2 shows no meaningful improvement
on the 40-point eval atlas over M1, the obvious next experiments are:

- Oversample pilot pairs 10–100× in `train/`.
- Add a separate short fine-tune phase on pilot-only data after main training.
- Use per-stmt DPO machinery (already on dpo-lora-clean) to do a pilot-only
  short run with a higher LR.

### 2. Reference = SFT

With `AdapterBaseReferenceConfig()` on marin-8b-instruct, DPO loss measures
divergence from SFT. This is identical to M1's setup. M2 will likely converge
to something close to M1 (same data + tiny perturbation + same reference).

### 3. Evaluation on the 40 atlas points

After training completes:
- Export M2 to HF format (auto via `steps_per_hf_export=200`).
- Run `bcg_probe_infer.py` on the 40 atlas prompts with `--target <M2>` at
  N=10.
- Score via gpt-5.1 batch (reasoning_effort=none), same as M1 N=10 gate.
- Compare M2 vs M1 per-point JSR + marginals.

(That eval is separate from the training submission; can be scripted morning
of.)
