# Belay+CPLM on TPU v4

This variant adds the [Copy-Paste Language Model](https://arxiv.org/abs/2610.19247v1)
head to Belay base16 d512. It uses the original Belay development branch
`ladderclimb_v1_batch14` (`358283f7`), with the qualified TPU execution changes
from `aef63f29`. The portable copy loss is shared with [PR #10230](https://github.com/marin-community/marin/pull/10230).
Historical July quality results in [issue #10229](https://github.com/marin-community/marin/issues/10229)
retain their July identity.

The frozen recipe uses 6 layers, 512 experts/top-8, ungated ReLU² experts,
KDA/MLA/Inkling attention, AttnRes, bigram embeddings, vocabulary 128256 and
sequence length 8192. Copy query/key matrices and gate/sink vectors add 525312
parameters; initialization uses an independent folded PRNG stream. The vocabulary
and COPY logits have softcap 10. The target marginal sums matching observed
positions in the same packed document, including the current token. The native
training mask excludes invalid next-token labels. Copy query tiles of 128 bound
score storage; arithmetic remains quadratic in sequence length. Generation and
KV-cache decoding are not implemented.

Training uses EP1 dropless on 16 physical TPU v4 chips, with FSDP weight gathers
and gradient reductions. The local scatter backend supports the ungated expert
projection. The QB histogram uses exact integer matrix counts at the local
8192-token × 512-expert shape, followed by the selected two-stage threshold
calculation. KDA state passes, projection gathers, column-sharded bigram lookup
and Muon TPU fallback preserve the measured execution path.

The logical optimizer batch is fixed at 16×8192; gradient accumulation is 1.
The MuonH/Adam recipe and learning-rate schedule remain 10980 updates, seed 0.
The [tokenizer SHA256 identities](tokenizer_identity.json) and
[July training/validation cache locations and weights](july_data.json) are checked in.
The tokenizer is `meta-llama/Meta-Llama-3.1-8B`; obtain its three tokenizer JSON
files from an authorized Hugging Face snapshot and verify the recorded hashes. These existing caches are in `us-central2`; provision access
before launching. This recipe intentionally pins regional cached inputs instead
of rebuilding or transferring the data. Supply the matching local tokenizer
folder on each worker; the entry point verifies all three tokenizer files.

## Run

Use the repository TPU environment (`uv sync --package marin-levanter --extra tpu`) and an already
configured TPU v4-32 distributed job, with one process per host. Iris/Levanter
handles distributed initialization. Supply online W&B credentials through the
job environment. Queue and task time limits are left to the caller; use Iris's
default preemption retry policy.

Run the non-training tracker check once on the remote submit host, outside Iris:

```bash
uv run --package marin-levanter --extra tpu python -m experiments.grug.moe_cplm.train_belay \
  --tokenizer /path/to/frozen-tokenizer --output /path/to/results \
  --checkpoint-path gs://marin-us-central2/PROJECT/checkpoints \
  --artifact-root gs://marin-us-central2/PROJECT/results \
  --run-name belay-cplm-d512 --wandb-entity ENTITY --wandb-project PROJECT \
  --logging-check
```

For Iris, stage the matching tokenizer under `tokenizer/` in the workspace and
submit the worker command from the repository root. The CLI infers four
co-scheduled hosts from v4-32:

```bash
iris --cluster marin job run --enable-extra-resources --tpu v4-32 \
  --cpu 4 --memory 48GB --disk 5GB --region us-central2 \
  --sync-package marin-levanter --extra tpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e JAX_PLATFORMS tpu,cpu -- \
  python -m experiments.grug.moe_cplm.train_belay \
  --tokenizer tokenizer --output /tmp/belay-results \
  --checkpoint-path gs://marin-us-central2/PROJECT/checkpoints \
  --artifact-root gs://marin-us-central2/PROJECT/results \
  --run-name belay-cplm-d512 --wandb-entity ENTITY --wandb-project PROJECT \
  --calibration
```

Use the worker command with `--calibration` in place of
`--logging-check` for 400 updates without evaluation or checkpoint writes.
Omit both flags for the full 10980-update schedule, with checkpointing and
validation every 1000 updates. Validation retains logical batch 256 and 2048
examples per domain, executed as physical microbatches of 16. The native port's
full training, checkpoint restore and validation endpoint have not been tested.

Each fresh process group broadcasts a new execution ID. W&B curves, checkpoint
paths and local/regional artifacts are separated by that ID. A restart begins
from initialization; it does not claim checkpoint resume. The native tracker
starts before optimizer updates. The first batch must pass a finite full-model
forward/backward check with nonzero copy-head gradients. Incremental native
loss logs are saved locally and uploaded every 60 seconds, including partial
curves after failures.

Calibration drains device work at updates 100 and 400 and divides 39,321,600
tokens by the maximum host elapsed time. Normal host callbacks remain included.
`training-window.json`, `native-metrics.jsonl`, `tracking.json`,
`resolved-config.json` and `full-model-preflight.json` are retained per execution.

## Measured provenance

These measurements use the frozen research harness with Marin `aef63f29`, before
this native CLI port. Each row completed 400 updates at B16×8192 on TPU v4-32;
the sustained window covers updates 100–399. All rows have seed 0.

| Execution | Sustained tokens/s | Median step tokens/s | Useful MFU | Max dropped fraction |
|---|---:|---:|---:|---:|
| EP16 control | 240044 | 256315 | 3.29% | 4.16% |
| EP4 native | 226789 | 239635 | 3.11% | 2.11% |
| EP1 dropless | 230953 | 232730 | 3.16% | 0 |
| EP1 exact histogram | 416823 | 424208 | 5.71% | 0 |
| EP1 exact histogram, fresh repeat | 418299 | 424206 | 5.73% | 0 |

Useful MFU uses 602,567,572.909 nominal model FLOPs per token and 4.4 PFLOP/s
peak for 16 TPU v4 chips; it excludes optimizer, rematerialization and padding
work. Native trainer MFU uses a different architecture ledger. EP16 is a
performance reference, with its own routing/drop behavior. No cross-EP gradient
matching is required. All 400 losses of the first optimized run exactly match
the unoptimized EP1 run; final NLL is 4.25661278. No validation loss was measured
in these calibrations, and they do not establish later schedule throughput.

[First optimized run](https://wandb.ai/marin-community/auto-understanding-arch/runs/mfu-ep1-hist-v2-a0ca4b4ede)
and [fresh repeat](https://wandb.ai/marin-community/auto-understanding-arch/runs/mfu-ep1-hist-repeat-v1-e40981c4c8)
contain the unsmoothed loss curves. Raw logs, profiles, failed attempts and
source hashes are indexed in the
[frozen run report](https://github.com/WhenWen/astra-d8-research/blob/beb6192b/research/belay-mfu-native-ep-result.md)
and [protocol](https://github.com/WhenWen/astra-d8-research/blob/116c07c1/research/belay-mfu-protocol.md), linked from [the performance issue](https://github.com/WhenWen/astra-d8-research/issues/236).
The independent native-port TPU calibration is pending; these rows do not
attribute the historical measurements to the port's new commit.

Run non-training numerical tests with:

```bash
uv run pytest experiments/grug/moe_cplm/test_head.py \
  lib/levanter/tests/grug/test_scatter_ungated.py \
  experiments/grug/fast_track/test_qb_balancing.py
```

The TPU histogram cases additionally require TPU v4. Copy loss checks cover
independent dense mixture values/gradients and causal/document isolation.
The scatter regression covers gated/ungated experts, FP32/BF16, concentrated
routing and masked tokens. These checks contain no optimizer updates.
