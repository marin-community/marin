# Mochi per-update weight movement: RL versus SFT

Measures how far one optimizer update moves Mochi's weights, per parameter group,
for the RL campaign (`.131` recipe) versus the SFT stage, from native checkpoints.

Feeds [marin #12438](https://github.com/marin-community/marin/issues/12438): whether
FP32-stored SFT rounds move weights much further per update than the campaign's RL.

## Method

For two checkpoints A and B exactly `n` optimizer updates apart, per sampled tensor slice:

- relative update `||W_b - W_a||_F / ||W_a||_F`, divided by `n`;
- median per-element `|Δ| / |W|` (strided subsample, pooled per group);
- fraction of elements whose BF16-rounded value changed (`round_bf16(W_b) != round_bf16(W_a)`).

Groups: attention projections; expert MLP banks; shared/dense MLP; router; norm scales;
embeddings and output projection (plus an `attention_conv_gate` extra). Every pair is
measured both `as-stored` and in `bf16` lands (both sides quantized to BF16 first).

Both sides of every pair are read element-aligned: Megatron master weights keep the HF
export orientation, so the reader transposes them into the Levanter orientation before
slicing. The calibration pair `calib-exp3-native-vs-export` (same step, FP32 native vs
its own BF16 export) validates orientation and measures pure BF16 export rounding.

## Checkpoint inventory (all reads read-only, all in cw-us-east-08a)

| id | URI | kind | step |
|---|---|---|---|
| hero-base-393491 | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-sft-initializers/hero-base393491-native/step-393491` | Levanter OCDBT FP32 | 393,491 |
| hero-sft-400407-export | `s3://marin-us-east-08a/marin/models/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1` | BF16 HF export | 400,407 |
| exp3-v5-native-2100 | `s3://marin-us-east-08a/marin/users/benfeuer/mochi-long-loop/long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/checkpoints/step-2100` | Levanter OCDBT FP32 | 2,100 |
| v5-sft2100-export | `.../long-loop-mochi-sft-exp3-v5-bankadam-128/sft/2026.11.24.128/exports/step-2100` | BF16 HF export | 2,100 |
| rl-533-step3 | `s3://marin-us-east-08a/tmp/ttl=14d/skyrl/.../rl-data-campaign/v3-mix-007/2026.11.26.533/checkpoints/global_step_3/policy` | Megatron DCP (FP32 masters) | 3 |
| rl-131-step33/34 | `.../long-loop/mochi-exp4-dspark-rl/2026.11.25.131/checkpoints/global_step_{33,34}/policy` | Megatron DCP | 33, 34 |
| rl-129-step24/25 | `.../long-loop/mochi-v5-exp3-rl/2026.11.25.129/checkpoints/global_step_{24,25}/policy` | Megatron DCP | 24, 25 |
| rl-719-step20/25 | `.../rl-data-campaign/v2-mix-033/2026.11.27.719/checkpoints/global_step_{20,25}/policy` | Megatron DCP | 20, 25 |
| rl-721-step5/10 | `.../rl-data-campaign/v2-mix-019/2026.11.27.721/checkpoints/global_step_{5,10}/policy` | Megatron DCP | 5, 10 |
| rl-717-step25/30 | `.../rl-data-campaign/v2-mix-009/2026.11.27.717/checkpoints/global_step_{25,30}/policy` | Megatron DCP | 25, 30 |
| rl-403-step1/2 | `.../long-loop/anchor-mochi-acceptance/2026.11.20.403/checkpoints/global_step_{1,2}/policy` | Megatron DCP | 1, 2 |
| rl-123-step9/22 | `.../long-loop/mochi-throughput-train-track1/2026.11.23.123/checkpoints/global_step_{9,22}/policy` | Megatron DCP | 9, 22 |

Pairs and their update counts are in `registry.PAIRS`. Notable deviations from the brief:

- The brief's first RL pair "SFT 400407 → RL step 3" is not exactly available: no
  400407-initialized run retains a step-3 checkpoint. Nearest anchors:
  `rl-400407-to-403s1/2` (steps 1/2) and `rl-400407-to-123s9` (step 9); those two runs
  differ from `.131` only in `max_grad_norm` (0.0 vs 0.5). The exact-`.131` step-3
  anchor is `rl-533-0to3`, whose initializer is the campaign's SFT export (v5-sft2100).
- No two hero-SFT FP32 checkpoints exist a known number of updates apart (the hero run
  retained only its final step, and the 08a copy is a BF16 export). The clean FP32 SFT
  pair is `sft-exp3-0to2100` (hero base → exp3-v5-bankadam final, the SFT stage that
  produced the campaign initializer). The hero stage is measured `sft-hero-0to6916`
  with the BF16 caveat, plus what BF16 can show on its own.

## Fixed sample (identical on both sides of every pair)

Layers every 8th plus the last: {0, 8, 16, 24, 32, 40, 47}. Experts 0..5 (one
contiguous bank group on both OCDBT chunk layouts). Expert layer groups 0-5, 24-29,
42-47. Expert-matrix area slab dim2 0..383 x dim3 0..767 (1/64 of each 3072x3072
expert matrix — the largest slab both OCDBT layouts bound in one chunk). Attention
rows/cols 0..47 (w_q rows, w_o cols, w_k/w_v rows 0..95). Embedding row bands
0-767 and 127488-128255. Router layers {0,1,16,17,32,33,46,47}. Norm vectors in full.

## Running

In region (CPU job on cw-us-east-08a, interactive priority):

```bash
# 1. Inventory: dump DCP key structure + build Megatron->canonical mappings
python experiments/active/mochi-update-magnitude/inventory.py --out s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude/inventory

# 2. Measurement
python experiments/active/mochi-update-magnitude/measure.py --stage rl
python experiments/active/mochi-update-magnitude/measure.py --stage sft
```

Local smoke (one tiny tensor, no full checkpoints through the Mac):

```bash
python experiments/active/mochi-update-magnitude/measure.py \
  --pairs sft-exp3-0to2100 --limit-tensors 1 --out /tmp/mum-smoke
```

## Outputs

`rows-<stamp>.jsonl` (one row per pair x tensor-slice x lands) and
`summary-<stamp>.json` (per pair x group aggregates plus the sample description),
under `s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude/results/`.
