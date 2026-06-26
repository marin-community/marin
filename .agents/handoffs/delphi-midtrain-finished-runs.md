# Delphi midtraining sweep — finished runs (for inference handoff)

> ## 🚨 CRITICAL — 1e20 RUNS IN THIS HANDOFF WERE TRAINED FROM A NON-DELPHI BASE 🚨
>
> **Discovered 2026-05-14.** The 12 "1e20" cells listed below were midtrained from `isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5` — a deprecated v5 isoflop ablation point, NOT a Delphi v6 compute-optimal model. Will Held (Delphi lead) confirmed the mismatch.
>
> Canonical Delphi 3e20 is `isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6` (d=2304 vs 2048, L=23 vs 21, v6 heuristic).
>
> **For inference / downstream use:** the 1e20 HF checkpoints below are valid models, but **do not refer to them as "Delphi 1e20."** Label them "v5-isoflop-3e20" or note the family mismatch. 1e21 and 1e22 entries below ARE valid Delphi runs.
>
> **Full post-mortem:** [`.agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md`](../ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md)

---

**Last updated:** 2026-05-05. **31 of 36 cells with usable HF checkpoints.** Remaining: 2 still training, 3 blocked on GCP capacity restoration.

## What this is

Each cell is a separate Delphi base model fine-tuned on a math-heavy mixture for ~20% of its pretrain budget. The sweep varies:
- **Scale** (1e20 / 1e21 / 1e22 pretrain FLOPs) — different base architectures, different starting checkpoints
- **Mix** (p33m67 / p50m50 / p67m33) — pretrain-replay : math weighting in the midtraining data
- **LR factor** (0.33 / 0.5 / 0.67 / 0.83) — multiplier on each base's pretrain peak LR

Each finished run wrote both raw Levanter checkpoints and HF-format checkpoints to GCS. **For inference, use the HF path.**

## Inference quick start

For any cell below, the inference checkpoint is:

```
gs://marin-us-east5/checkpoints/{run_name}/hf/step-{final_step}/
```

Where `final_step` is `9412` (1e20), `4410` (1e21), or `7646` (1e22) — Levanter's last-step is `num_train_steps - 1`.

```bash
gsutil -m cp -r gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a/hf/step-7646 ./local-ckpt
```

**Important load recipe** (these checkpoints have known metadata bugs — see Caveats below):

```python
from transformers import AutoTokenizer, AutoConfig, Qwen3ForCausalLM
import json

# 1. Strip the bad rope_scaling field BEFORE transformers.AutoConfig validates
with open("./local-ckpt/config.json", "r") as f:
    cfg = json.load(f)
cfg.pop("rope_scaling", None)               # see Caveat #2
cfg["architectures"] = ["Qwen3ForCausalLM"] # see Caveat #1
with open("./local-ckpt/config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# 2. Load the model as Qwen3 explicitly (don't trust AutoModel)
model = Qwen3ForCausalLM.from_pretrained("./local-ckpt", torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained("./local-ckpt")  # tokenizer files are correct (Llama-3.1)
```

For **vLLM serving**, pass `hf_overrides` at engine init (and pre-strip rope_scaling on a local config copy). See `experiments/evals/exp_eval_delphi_midtrain_math500.py` on the `patch_vllm` branch for a working reference.

## Architecture per scale

| Scale | hidden_dim | seq_len | global_batch | peak_lr (base) | midtrain steps | midtrain tokens |
|---|---|---|---|---|---|---|
| **1e20** | 2048 | 4096 | 128 | 4.483e-3 | 9,413 | 4.94 B |
| **1e21** | 2560 | 4096 | 512 | 7.425e-3 | 4,411 | 9.25 B |
| **1e22** | 3840 | 4096 | 1024 | 7.232e-3 | 7,647 | 32.07 B |

Each LR factor multiplies the base peak_lr (e.g. 1e22 lr=0.5 → peak_lr = 3.616e-3). Tokenizer: `meta-llama/Meta-Llama-3.1-8B`. Optimizer: AdamH.

## Midtraining data mixture

| Label | Pretrain replay | Math share | Math source |
|---|---|---|---|
| `p33m67` | 33% (Nemotron mix) | 67% | NemotronCC-math-v1 4plus filter |
| `p50m50` | 50% (Nemotron mix) | 50% | NemotronCC-math-v1 4plus filter |
| `p67m33` | 67% (Nemotron mix) | 33% | NemotronCC-math-v1 4plus filter |

12,500 sequences held out from training for validation (deterministic Feistel index split).

## Wandb project

[`https://wandb.ai/marin-community/delphi-midtraining`](https://wandb.ai/marin-community/delphi-midtraining)

---

## Finished runs (31 cells with usable HF checkpoint)

### 1e20 — 12/12 ✅

```
p33m67 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p33m67-4p94b-lr0.33-307237
               gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.33-307237/hf/step-9412/
p33m67 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p33m67-4p94b-lr0.5-2004f8
               gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.5-2004f8/hf/step-9412/
p33m67 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p33m67-4p94b-lr0.67-7c32da
               gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.67-7c32da/hf/step-9412/
p33m67 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p33m67-4p94b-lr0.83-2a22e0
               gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.83-2a22e0/hf/step-9412/
p50m50 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p50m50-4p94b-lr0.33-9a74fa
               gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.33-9a74fa/hf/step-9412/
p50m50 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p50m50-4p94b-lr0.5-3475fa
               gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.5-3475fa/hf/step-9412/
p50m50 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p50m50-4p94b-lr0.67-554fb6
               gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.67-554fb6/hf/step-9412/
p50m50 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p50m50-4p94b-lr0.83-95e10d
               gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.83-95e10d/hf/step-9412/
p67m33 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p67m33-4p94b-lr0.33-590ea1
               gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.33-590ea1/hf/step-9412/
p67m33 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p67m33-4p94b-lr0.5-9e1229
               gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.5-9e1229/hf/step-9412/
p67m33 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p67m33-4p94b-lr0.67-64a9c5
               gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.67-64a9c5/hf/step-9412/
p67m33 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e20-p67m33-4p94b-lr0.83-1965f3
               gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.83-1965f3/hf/step-9412/
```

### 1e21 — 10/12 ✅ (2 still training, see In-Progress below)

```
p33m67 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p33m67-9p25b-lr0.33-58ebcb
               gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.33-58ebcb/hf/step-4410/
p33m67 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p33m67-9p25b-lr0.5-efbc63
               gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.5-efbc63/hf/step-4410/
p33m67 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p33m67-9p25b-lr0.67-9cf8da
               gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.67-9cf8da/hf/step-4410/
p50m50 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p50m50-9p25b-lr0.33-bccff4
               gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.33-bccff4/hf/step-4410/
p50m50 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p50m50-9p25b-lr0.5-973c46
               gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.5-973c46/hf/step-4410/
p50m50 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p50m50-9p25b-lr0.67-7e82b3
               gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.67-7e82b3/hf/step-4410/
p50m50 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p50m50-9p25b-lr0.83-f9edd2
               gs://marin-us-east5/checkpoints/delphi-1e21-p50m50-9p25b-lr0.83-f9edd2/hf/step-4410/
p67m33 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64
               gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/hf/step-4410/
p67m33 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p67m33-9p25b-lr0.5-114e49
               gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49/hf/step-4410/
p67m33 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e21-p67m33-9p25b-lr0.83-a1a261
               gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.83-a1a261/hf/step-4410/
```

### 1e22 — 9/12 ✅ + 1 with HF saved despite wandb=crashed = 10 total usable

```
p33m67 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p33m67-32p07b-lr0.33-e9132105
               gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.33-e9132105/hf/step-7646/
p33m67 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d
               gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d/hf/step-7646/
p33m67 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7
               gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7/hf/step-7646/
p50m50 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p50m50-32p07b-lr0.33-c43ada
               gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.33-c43ada/hf/step-7646/
               (wandb shows state=crashed because iris was killed during finalize; HF FINAL ckpt is fully saved)
p50m50 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p50m50-32p07b-lr0.83-3c9f70
               gs://marin-us-east5/checkpoints/delphi-1e22-p50m50-32p07b-lr0.83-3c9f70/hf/step-7646/
p67m33 lr0.33  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7
               gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf/step-7646/
p67m33 lr0.5   https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a
               gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a/hf/step-7646/
p67m33 lr0.67  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p67m33-32p07b-lr0.67-3c17740e
               gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.67-3c17740e/hf/step-7646/
p67m33 lr0.83  https://wandb.ai/marin-community/delphi-midtraining/runs/delphi-1e22-p67m33-32p07b-lr0.83-d35daa
               gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.83-d35daa/hf/step-7646/
```

---

## In progress (2 cells)

These will land at `gs://marin-us-east5/checkpoints/{run_name}/hf/step-{final}/` once they finish:

```
1e21 p33m67 lr0.83  delphi-1e21-p33m67-9p25b-lr0.83-0cb048    ETA: ~30 min   (98% done on v5p-64)
1e21 p67m33 lr0.67  delphi-1e21-p67m33-9p25b-lr0.67-ecbd27    ETA: ~6-10h    (82% done on v5p-32, resumed from step-2646)
```

Re-run the wandb URL/HF path lookup after these complete:

```bash
gcloud storage ls gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.83-0cb048/hf/
gcloud storage ls gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/hf/
```

---

## Unfinished — capacity blocked (3 cells)

These have partial checkpoints preserved but stalled when GCP wholesale-reclaimed v5p-256/128/512 from iris on 2026-05-04 ~22:11 UTC. Resume when capacity returns.

| Cell | Last permanent ckpt | Last temp ckpt | % done |
|---|---|---|---|
| 1e22 p33m67 lr0.83 | `delphi-1e22-p33m67-32p07b-lr0.83-78fd44/checkpoints/step-6876/` | step-7182 | ~94% |
| 1e22 p50m50 lr0.5 | `delphi-1e22-p50m50-32p07b-lr0.5-ecfa99/checkpoints/step-4584/` | step-5034 | ~66% |
| 1e22 p50m50 lr0.67 | `delphi-1e22-p50m50-32p07b-lr0.67-e78260/checkpoints/step-3056/` | step-3649 | ~48% |

Resume command pattern (when v5p-256 returns):

```bash
-e MIDTRAIN_RESUME_OUTPUT_PATH "gs://marin-us-east5/checkpoints/<run_name>"
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP "<perm_step>"
-e MIDTRAIN_SELECT_BASE "1e22-v5"
-e MIDTRAIN_SELECT_LR "<lr>"
-e MIDTRAIN_MIX_NAME "<mix>"
-e MIDTRAIN_TPU_TYPE "v5p-256"
```

---

## Caveats for the labmate

### 1. `architectures` says Llama, weights are Qwen3

The HF `config.json` has `architectures: ["LlamaForCausalLM"]` due to marin PR #3092 (Qwen3 export bug, still open). The actual weights are Qwen3 — they include `q_norm` / `k_norm` keys, vocab=128256, no attention biases.

**Don't use `AutoModelForCausalLM`** — it'll trust the bad metadata. Use `Qwen3ForCausalLM.from_pretrained(...)` directly, OR override the field before loading (see Inference quick-start above).

### 2. `rope_scaling.original_max_position_embeddings=8192 > max_position_embeddings=4096`

The saved configs ship a llama3-style `rope_scaling` block where `original_max_position_embeddings` is GREATER than `max_position_embeddings` — backwards from what transformers expects. transformers' `AutoConfig.from_pretrained` raises `ValueError` on load.

Workaround: **strip `rope_scaling` from a local copy of `config.json` before loading.** The model was trained at native 4096 context with no RoPE extension, so removing the field is safe.

### 3. Tokenizer is correct (Llama-3.1)

`tokenizer.json` and `tokenizer_config.json` in each HF dir are real Llama-3.1 tokenizer files (vocab 128256). Don't override these.

### 4. Suffix hashes are config hashes, not wandb auto-IDs

The 6-8 char hex suffix on each run name (e.g. `f60cb12a`, `ab4e64`) is the marin StepSpec config hash for that run, set as the wandb `run_id` by the experiment script. They're stable per-config; resumes preserve them.

### 5. Eval losses

Recorded in wandb under `eval/loss` per validation set. Use those rather than re-running eval if you just want validation comparisons.

---

## Source code reference

- Sweep definition: `experiments/exp_delphi_math_10b_midtrain.py`
- Mix definitions: `experiments/midtraining_mixes.py`
- Logbook (full sweep history): `.agents/logbooks/midtraining_delphi.md`
- Iris placement-bug postmortem (relevant for resumes): `.agents/ops/iris_placement_bug.md`
- vLLM eval recipe (architectures + rope_scaling overrides applied at runtime): `experiments/evals/exp_eval_delphi_midtrain_math500.py` on the `patch_vllm` branch
