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

Dense at 20 tokens/active-param, MoE at 60; recorded at batch 128 on 1 node (8×H100); base LR. bpb is
macro bits-per-byte (lower = better).

| size | variant | TPP | steps | tokens | FLOPs | MFU | Paloma bpb | uncheat bpb | runtime |
|------|---------|----:|------:|-------:|------:|----:|-----------:|------------:|--------:|
| d512  | dense | 20 |    690 | 0.36B | 3.2e17 | 33.1% | 1.516 | 1.239 | 3.7m |
| d512  | moe   | 60 |  2,385 | 1.25B | 7.6e17 |  7.6% | 1.309 | 1.008 | 14.6m |
| d768  | dense | 20 |  2,040 | 1.07B | 2.1e18 | 52.2% | 1.359 | 1.062 | 9.4m |
| d768  | moe   | 60 |  6,930 | 3.63B | 4.2e18 |  9.4% | 1.193 | 0.889 | 1.0 hr |
| d1024 | dense | 20 |  5,400 | 2.83B | 1.2e19 | 70.0% | 1.242 | 0.941 | 36.1m |
| d1024 | moe   | 60 | 18,180 | 9.53B | 2.0e19 | 11.0% | 1.100 | 0.795 | 4.5 hr |
| d1280 | dense | 20 |  9,800 | 5.14B | 3.8e19 | 82.4% | _est_ | _est_ | ~1.5 hr |
| d1280 | moe   | 60 | 32,813 | 17.2B | 5.6e19 | 11.9% | _est_ | _est_ | ~12.3 hr |

d1280 rows are estimated from 20-step MFU probes (bpb pending a full run). The d1024/d1280 rows
predate two config changes now in the launcher — `local_kv_heads=2` and a baseline batch of 256 — so
their steps/tokens/FLOPs no longer match the current defaults and the bpb needs a re-run; d512/d768
(batch 128, unchanged) are current.

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
irun dense-d768 --run-id dense-d768 --size d768 --dense \
  --no-save-checkpoints --version 2026.09.17
```

MoE (data-match baseline; bump the batch — steps halve to hold tokens):

```bash
irun moe-d768 --run-id moe-d768 --size d768 --batch-size 256 \
  --no-save-checkpoints --version 2026.09.17
```

MFU probe (any size, quick — explicit short budget):

```bash
irun probe-d1280 --run-id probe-d1280 --size d1280 --num-steps 20 \
  --no-eval --no-save-checkpoints --version 2026.09.17
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
| `--no-save-checkpoints` | no checkpoints (throughput runs) |

Results land in W&B `marin-community/marin_moe`; eval bpb keys are `eval/paloma/macro_bpb`,
`eval/uncheatable_eval/macro_bpb` (MoE dropless eval logs under the normal `eval/` prefix).
</content>
