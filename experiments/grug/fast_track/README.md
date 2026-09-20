# fast_track — dense vs MoE scaling ladder (H100, 16k BPE)

Self-contained grug variant for the 16k-vocab (BPE) dense-vs-MoE comparison + MFU study on 8×H100.
Entry point is [`launch.py`](launch.py). Data: the in-region `v16384_shuf` flat cache — a document
shuffle (see [`../../datakit/shuffle_cache.py`](../../datakit/shuffle_cache.py)) of the tokenized
`v16384` cache built by the Datakit pipeline (see [`../../datakit/hero_data.py`](../../datakit/hero_data.py)
for the source/tokenizer/version pins). Eval: Paloma + uncheatable bits-per-byte (in-region,
detokenized — see [`../../datakit/paloma_detok.py`](../../datakit/paloma_detok.py)).

## Files

| file | contents |
|------|----------|
| [`launch.py`](launch.py) | ladder rungs, budget resolution (`--match`), Iris/W&B wiring |
| [`model.py`](model.py) | the transformer: attention, GatedNorm, SConv, QB-routed MoE |
| [`train.py`](train.py) | trainer/eval/loss wiring and runtime (XLA) defaults |
| [`optimizer.py`](optimizer.py) | MuonH optimizer config: LR groups + hyperball step |
| [`grugmuon_stacked.py`](grugmuon_stacked.py) | Newton-Schulz orthogonalization (Muon direction) |
| [`adamh.py`](adamh.py) | AdamH scale transform (the `adamh` LR group) |
| [`heuristic.py`](heuristic.py) | compute-scaling LR / beta2 / epsilon fit |
| [`router_metrics.py`](router_metrics.py) | routing-stats telemetry (logging-only) |

## Results

| size | variant | TPP | batch | steps | tokens | FLOPs | MFU | Paloma loss | Paloma bpb | uncheat bpb | runtime |
|------|---------|----:|------:|------:|-------:|------:|----:|------------:|-----------:|------------:|--------:|
| d512  | dense | 20 | 128 |    690 | 0.36B | 9.0e16 | 15.8% | 3.676 | 1.520 | 1.243 | 3.7m |
| d512  | moe   | 60 | 128 |  2,385 | 1.25B | 3.3e17 |  7.8% | 3.156 | 1.308 | 1.008 | 14.0m |
| d768  | dense | 20 | 128 |  2,040 | 1.07B | 6.3e17 | 20.2% | 3.283 | 1.361 | 1.063 | 9.2m |
| d768  | moe   | 60 | 128 |  6,930 | 3.63B | 2.3e18 | 10.3% | 2.872 | 1.193 | 0.888 | 54.4m |
| d1024 | dense | 20 | 256 |  2,760 | 2.89B | 3.9e18 | 26.7% | 3.006 | 1.248 | 0.945 | 34.8m |
| d1024 | moe   | 60 | 256 |  9,270 | 9.72B | 1.4e19 | 13.9% | 2.633 | 1.096 | 0.793 | 3.8 hr |
| d1280 | dense | 20 | 256 |  4,988 | 5.23B | 1.2e19 | 28.4% | 2.847 | 1.183 | 0.881 | 1.5 hr |
| d1280 | moe   | 60 | 256 | 16,669 | 17.5B | 4.2e19 | 15.4% | 2.504 | 1.044 | 0.741 | 10.0 hr |

## Tokenizer impact (16k vs 128k)

Same MoE geometry and token budget, swapping the 16k BPE tokenizer for the 128k Marin (llama3-family)
tokenizer. bpb is byte-normalized so it compares fairly across tokenizers; per-token loss does not
(the 128k tokenizer packs more bytes per token, so higher loss at similar bpb).

| size | tokenizer | Paloma loss | Paloma bpb | uncheat bpb |
|------|-----------|------------:|-----------:|------------:|
| d512 | 16k  | 3.156 | 1.308 | 1.008 |
| d512 | 128k | 3.656 | 1.309 | 0.996 |
| d768 | 16k  | 2.872 | 1.193 | 0.888 |
| d768 | 128k | 3.306 | 1.187 | 0.870 |

Both runs train on the same number of tokens, but the 16k tokenizer compresses ~12% worse than the
128k, so at equal token budget the 16k run covers ~12% fewer bytes of text. Even with that data
disadvantage it lands ~neutral on Paloma bpb (+0.001 at d512, −0.006 at d768) and only slightly behind
on uncheatable bpb (−0.012 / −0.018).

## Scaling law

Fitting `L(C) = L∞ + A·C^(−α)` to Paloma macro loss over the four rungs (C = total training FLOPs), with
the irreducible floor pinned at **L∞ = 1.2**:

| variant | fit | R² | α |
|---------|-----|---:|--:|
| MoE   | `L = 1.2 + 56.76·C^(−0.0835)` | 0.99979 | 0.0835 |
| dense | `L = 1.2 + 64.53·C^(−0.0836)` | 0.99917 | 0.0836 |

![Paloma scaling law](scaling_law.png)

With a shared floor the two exponents are essentially identical (α ≈ 0.084 for both), so dense and MoE
scale **in parallel** and the gap is a constant amplitude offset (A: 64.5 dense vs 56.8 MoE).
**Compute efficiency:** the MoE recipe (60 TPP) reaches the same Paloma loss as the compute-optimal
dense recipe (20 TPP) with **~4.3× less compute, roughly constant across the ladder** (parallel lines,
so the multiple does not widen with scale). This is a recipe-vs-recipe comparison — dense at its
compute-optimal 20 TPP, MoE at 60 TPP — so it bundles the architecture gain with the 3× longer MoE
token schedule and is not an iso-TPP architecture measurement. Caveat: with the floor fixed, two free
parameters (A, α) fit four points; trust α and the ordering, not the L∞ value or extrapolations far
past ~5e19 FLOPs.

## Launch commands

Set `$WANDB_API_KEY` in your shell.

Common wrapper:

Usage is `irun <job-name> <launch-args…>`. It targets `$IRIS_CLUSTER` (default `cw-rno2a`); both it
and `cw-us-east-02a` are 8×H100 clusters: `IRIS_CLUSTER=cw-us-east-02a irun …`.

```bash
irun() { uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \
  --target-cluster "${IRIS_CLUSTER:-cw-rno2a}" --priority interactive --job-name "$1-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_PROJECT marin_moe \
  -- python -m experiments.grug.fast_track.launch "${@:2}" --run; }
```

Pick a size and variant; the budget defaults to **data-matching** that variant's baseline (dense at
20 TPP, MoE at 60 TPP) at the rung's baseline batch (128 for d512/d768, 256 for d1024/d1280). Steps
are derived automatically. Use `--match compute` to FLOP-match instead, `--batch-size` to change the
batch (steps rescale to hold the match), or `--num-steps` to set the count explicitly.

Dense (data-match baseline):

```bash
irun dense-d768 --run-id dense-d768 --size d768 --dense --version 2026.09.17
```

MoE (data-match baseline; bump the batch — steps halve to hold tokens):

```bash
irun moe-d768 --run-id moe-d768 --size d768 --batch-size 256 --version 2026.09.17
```

MFU probe (any size, quick — explicit short budget):

```bash
irun probe-d1280 --run-id probe-d1280 --size d1280 --num-steps 20 --no-eval --version 2026.09.17
```

### Useful flags (all on `launch.py`)

| flag | effect |
|------|--------|
| `--run-id` | **required** run identifier for artifact + W&B names |
| `--size` | **required** `d512` / `d768` / `d1024` / `d1280` |
| `--dense` | dense 3×hidden SwiGLU, no MoE (expert=1, normal eval) |
| `--match` | `data` (default) tokens-match or `compute` FLOP-match the variant baseline |
| `--batch-size` | override the rung's baseline batch (steps rescale to hold the match) |
| `--num-steps N` | set the step budget explicitly (ignores `--match`) |
| `--no-eval` | skip eval (clean MFU probes) |
| `--save-checkpoints` | save a permanent final checkpoint to S3 (off by default) |

Results land in W&B `marin-community/marin_moe`; eval bpb keys are `eval/paloma/macro_bpb`,
`eval/uncheatable_eval/macro_bpb` (MoE dropless eval logs under the normal `eval/` prefix).
</content>
