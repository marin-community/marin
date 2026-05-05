# Delphi midtraining sweep — finished runs (for inference handoff)

**Last updated:** 2026-05-04. **22 of 36 cells finished.** The remaining 14 are still training; see "in-flight" section at the bottom.

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

The HF directory contains:
- `config.json` — model architecture (hidden_size, num_hidden_layers, num_attention_heads, etc.)
- `model-NNNNN-of-NNNNN.safetensors` — sharded weights
- `tokenizer*.json` — Meta-Llama-3.1-8B tokenizer artifacts

To download one cell locally:

```bash
gsutil -m cp -r gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a/hf/step-7646 ./local-ckpt
```

Then load with `transformers.AutoModelForCausalLM.from_pretrained("./local-ckpt")` (it's a Qwen3-architecture variant — needs a recent enough `transformers`; verify `config.json:model_type` and use `trust_remote_code=True` if needed).

## Architecture per scale

| Scale | hidden_dim | seq_len | global_batch | peak_lr (base) | midtrain steps | midtrain tokens |
|---|---|---|---|---|---|---|
| **1e20** | 2048 | 4096 | 128 | 4.483e-3 | 9,413 | 4.94 B |
| **1e21** | 2560 | 4096 | 512 | 7.425e-3 | 4,411 | 9.25 B |
| **1e22** | 3840 | 4096 | 1024 | 7.647 | 32.07 B |

Each midtrain LR factor multiplies the base peak_lr. So for 1e22 lr0.5 → peak_lr = 0.5 × 7.232e-3 = 3.616e-3.

Tokenizer: `meta-llama/Meta-Llama-3.1-8B` for all scales. Optimizer: AdamH (Adam + heuristic schedule from `experiments.scaling_law_sweeps.completed_adamh.completed_adamh_heuristic`).

## Midtraining data mixture (per mix label)

| Label | Pretrain replay | Math share | Math source |
|---|---|---|---|
| `p33m67` | 33% (Nemotron mix) | 67% | NemotronCC-math-v1 4plus filter |
| `p50m50` | 50% (Nemotron mix) | 50% | NemotronCC-math-v1 4plus filter |
| `p67m33` | 67% (Nemotron mix) | 33% | NemotronCC-math-v1 4plus filter |

All mixes use the same math source (`HIGHQUALITY_NEMO_MATH_KEY`); only the weight differs. Held-out validation slice (12,500 sequences) carved out via deterministic Feistel index remap; no train/val leak.

## Wandb

**Project:** [`marin-community/delphi-midtraining`](https://wandb.ai/marin-community/delphi-midtraining)

To find a specific run, paste its `run_name` (the `delphi-{scale}-{mix}-{budget}-lr{lr}-{hash}` string from the table below) into the wandb search bar. Each run logs train/loss, eval/loss, gradient norms, and learning-rate schedule per step.

## Finished runs (22 cells)

### 1e20 (12/12 done — base = `isoflop-3e+20-d2048-L21`, hidden=2048, batch=128)

| Mix | LR factor | Effective peak_lr | Run name | HF checkpoint |
|---|---|---|---|---|
| p33m67 | 0.33 | 1.479e-3 | `delphi-1e20-p33m67-4p94b-lr0.33-307237` | `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.33-307237/hf/step-9412/` |
| p33m67 | 0.5  | 2.242e-3 | `delphi-1e20-p33m67-4p94b-lr0.5-2004f8`  | `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.5-2004f8/hf/step-9412/` |
| p33m67 | 0.67 | 3.004e-3 | `delphi-1e20-p33m67-4p94b-lr0.67-7c32da` | `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.67-7c32da/hf/step-9412/` |
| p33m67 | 0.83 | 3.721e-3 | `delphi-1e20-p33m67-4p94b-lr0.83-2a22e0` | `gs://marin-us-east5/checkpoints/delphi-1e20-p33m67-4p94b-lr0.83-2a22e0/hf/step-9412/` |
| p50m50 | 0.33 | 1.479e-3 | `delphi-1e20-p50m50-4p94b-lr0.33-9a74fa` | `gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.33-9a74fa/hf/step-9412/` |
| p50m50 | 0.5  | 2.242e-3 | `delphi-1e20-p50m50-4p94b-lr0.5-3475fa`  | `gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.5-3475fa/hf/step-9412/` |
| p50m50 | 0.67 | 3.004e-3 | `delphi-1e20-p50m50-4p94b-lr0.67-554fb6` | `gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.67-554fb6/hf/step-9412/` |
| p50m50 | 0.83 | 3.721e-3 | `delphi-1e20-p50m50-4p94b-lr0.83-95e10d` | `gs://marin-us-east5/checkpoints/delphi-1e20-p50m50-4p94b-lr0.83-95e10d/hf/step-9412/` |
| p67m33 | 0.33 | 1.479e-3 | `delphi-1e20-p67m33-4p94b-lr0.33-590ea1` | `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.33-590ea1/hf/step-9412/` |
| p67m33 | 0.5  | 2.242e-3 | `delphi-1e20-p67m33-4p94b-lr0.5-9e1229`  | `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.5-9e1229/hf/step-9412/` |
| p67m33 | 0.67 | 3.004e-3 | `delphi-1e20-p67m33-4p94b-lr0.67-64a9c5` | `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.67-64a9c5/hf/step-9412/` |
| p67m33 | 0.83 | 3.721e-3 | `delphi-1e20-p67m33-4p94b-lr0.83-1965f3` | `gs://marin-us-east5/checkpoints/delphi-1e20-p67m33-4p94b-lr0.83-1965f3/hf/step-9412/` |

### 1e21 (4/12 done — base = `1e21-v5`, hidden=2560, batch=512)

| Mix | LR factor | Effective peak_lr | Run name | HF checkpoint |
|---|---|---|---|---|
| p33m67 | 0.33 | 2.450e-3 | `delphi-1e21-p33m67-9p25b-lr0.33-58ebcb` | `gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.33-58ebcb/hf/step-4410/` |
| p33m67 | 0.5  | 3.713e-3 | `delphi-1e21-p33m67-9p25b-lr0.5-efbc63`  | `gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.5-efbc63/hf/step-4410/` |
| p33m67 | 0.67 | 4.975e-3 | `delphi-1e21-p33m67-9p25b-lr0.67-9cf8da` | `gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.67-9cf8da/hf/step-4410/` |
| p67m33 | 0.33 | 2.450e-3 | `delphi-1e21-p67m33-9p25b-lr0.33-ab4e64` | `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64/hf/step-4410/` |

### 1e22 (6/12 done — base = `1e22-v5`, hidden=3840, batch=1024)

| Mix | LR factor | Effective peak_lr | Run name | HF checkpoint |
|---|---|---|---|---|
| p33m67 | 0.33 | 2.387e-3 | `delphi-1e22-p33m67-32p07b-lr0.33-e9132105` | `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.33-e9132105/hf/step-7646/` |
| p33m67 | 0.5  | 3.616e-3 | `delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d`  | `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.5-0eeca70d/hf/step-7646/` |
| p33m67 | 0.67 | 4.845e-3 | `delphi-1e22-p33m67-32p07b-lr0.67-54770ae7` | `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7/hf/step-7646/` |
| p67m33 | 0.33 | 2.387e-3 | `delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7` | `gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.33-4e8cc7a7/hf/step-7646/` |
| p67m33 | 0.5  | 3.616e-3 | `delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a`  | `gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.5-f60cb12a/hf/step-7646/` |
| p67m33 | 0.67 | 4.845e-3 | `delphi-1e22-p67m33-32p07b-lr0.67-3c17740e` | `gs://marin-us-east5/checkpoints/delphi-1e22-p67m33-32p07b-lr0.67-3c17740e/hf/step-7646/` |

## Still in-flight (will finish over next 12-30h)

These cells are not yet done. Once finished they'll write to the same `gs://marin-us-east5/checkpoints/{run_name}/hf/step-{final}/` pattern, and you can update this doc by running:

```bash
gcloud storage ls "gs://marin-us-east5/checkpoints/delphi-{scale}-{mix}-{budget}-lr{lr}-*"
```

| Scale | Mix | LR | Status |
|---|---|---|---|
| 1e21 | p33m67 | 0.83 | pending v5p-128 |
| 1e21 | p50m50 | 0.33 | running v5p-128 |
| 1e21 | p50m50 | 0.5  | running v5p-128 |
| 1e21 | p50m50 | 0.67 | running v5p-128 |
| 1e21 | p50m50 | 0.83 | running v5p-128 |
| 1e21 | p67m33 | 0.5  | pending v5p-128 (resume from `114e49`) |
| 1e21 | p67m33 | 0.67 | pending v5p-128 (resume from `ecbd27`) |
| 1e21 | p67m33 | 0.83 | running v5p-128 |
| 1e22 | p33m67 | 0.83 | running v5p-256 |
| 1e22 | p50m50 | 0.33 | running v5p-256 |
| 1e22 | p50m50 | 0.5  | pending v5p-256 (resume from `ecfa99`) |
| 1e22 | p50m50 | 0.67 | pending v5p-256 (resume from `e78260`) |
| 1e22 | p50m50 | 0.83 | running v5p-256 |
| 1e22 | p67m33 | 0.83 | running v5p-256 |

## Source code reference

- Sweep definition: `experiments/exp_delphi_math_10b_midtrain.py`
- Mix definitions: `experiments/midtraining_mixes.py`
- Base model architecture builder: `experiments/scaling_law_sweeps/completed_adamh.completed_adamh_heuristic._build_model_config(hidden_size, seq_len)`
- Logbook: `.agents/logbooks/midtraining_delphi.md`

## Caveats for the labmate

1. **Tokenizer mismatch alarm:** the base checkpoints were pretrained with NemotronCC tokenization (which uses Llama 3.1 tokenizer). Don't tokenize inference inputs with a different tokenizer.
2. **Checkpoint format:** the `hf/` directory is HuggingFace `safetensors` shards. Use `transformers.AutoModelForCausalLM.from_pretrained(...)`. The `checkpoints/step-N/` dirs are Levanter raw format — only useful if you want to resume training, not for inference.
3. **Wandb fragmentation:** for the 1e22 p67m33 cells (`-4e8cc7a7`, `-f60cb12a`, `-3c17740e`), the wandb project shows TWO rows per cell — one fossil (~3 min, no checkpoint) and one live (full training). The live one is the row with significant runtime (~10+ hours). The HF checkpoints listed above are the correct ones; ignore the fossils.
4. **Eval losses are recorded in wandb** under `eval/loss` per validation set. Use those rather than re-running eval if you just want validation comparisons.
