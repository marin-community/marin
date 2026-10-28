# Belay Ckpt 1 on TPU

This variant reproduces the d512 Ckpt 1 recipe from upstream
`8415c3ce8411051c1e22bc13184c272b88ec2b6c`. It has six layers, 384 ReLU²
experts with top-10 routing, KDA local attention, MLA/Inkling global attention,
AttnRes and value embeddings. Ckpt 1 has no copy head. The separate
[Belay+CPLM baseline](../moe_cplm/README.md) retains its original recipe and
copy head.

The selected profile is `kda-owner-causal-sort-v1`. Its source implementation
completed the 3,634-update suffix from checkpoint 5,187 through update 8,820
and all 23 evaluation domains. Three synchronized 300-update windows measured
314.353 ms/update and 8.3620% useful MFU on v4-32. The measurement, loss curves
and limitations are in [the result record](results/2026-10-26.md). These are
measurements of source commit `22db0f03a73c5bc101d23e514ae3feb111e4a8a0`;
consolidation checks on this PR are recorded separately there.

## Execution choices

`tpu_kernel_profiles.py` binds the trainer and checkpoint diagnostic to the
same named profiles. Select the profile explicitly: the native trainer's
default remains `source`.

| Component | Selected implementation |
|---|---|
| Expert placement | Keep tokens local; gather compute weights while master parameters and optimizer state stay sharded |
| Capacity | Sixteen pools of 24 experts; preserve expert-major, sender-major acceptance order and source capacity factor 1.05 |
| Routing statistics | Factorized integer histograms and interval capacity layout |
| Expert derivatives | Saved activations, two row partitions, grouped products for input and weight gradients |
| Dispatch/combine | Streaming route accumulation and scalar sorting; avoid complete wide token-slot restores |
| Weight derivatives | Exchange source-dtype contributions to their owner, then sum in FP32 |
| KDA | Chunk 64, four saved triangular-solve groups, fused preparation/recurrence, HIGH state products |
| Global attention | Causal buckets with saved backward activations |

Training retains the source routing and drop semantics. Evaluation uses the
existing dropless scatter path. The tokenizer with a 16,384-token vocabulary, data order,
batch 128 × sequence 4,096, seed 0 and 8,821-update MuonH+AdEMAMix schedule
are fixed by `belay_ckpt1.py` and checked by `validate_recipe`. The tokenizer
files are checksum-verified before use.

The recipe/model/optimizer snapshot is intentionally separate from
`fast_track`: replacing that directory would alter the older CPLM baseline.
The snapshot comes from the Ckpt 1 source line, with the TPU changes listed
above. GPU-only transport extensions are rejected by this variant. Applying
a profile changes only this variant's process-local operators; it does not
patch Levanter's shared MoE implementation. Use a fresh process per profile.

## Run and validate

Run on a four-host v4-32 environment with `uv sync --extra tpu` and the
checked-in [lockfile](../../../uv.lock),
GCS access to the frozen caches, and online W&B credentials. Every fresh
execution gets a separate W&B identity. A resumed run must restore complete
model, optimizer, router and step state; do not resume a curve merely because
the run name matches.

```bash
export JAX_PLATFORMS=tpu,cpu
export XLA_FLAGS=--xla_allow_excess_precision=false
export LIBTPU_INIT_ARGS='--xla_enable_async_all_reduce=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_all_to_all=false --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion=false --xla_tpu_enable_latency_hiding_scheduler=true'
export BELAY_MEASURE_MFU=1
export IRIS_OUTPUT_DIR=/path/to/attempt-output
python -m experiments.grug.belay_tpu.belay_ckpt1_tpu \
  --run-id YOUR_UNIQUE_RUN_ID --output-path gs://YOUR_BUCKET/YOUR_UNIQUE_RUN_ID \
  --tpu v4-32 --seed 0 --local-experts \
  --kernel-profile kda-owner-causal-sort-v1 --pipeline-depth 1 \
  --config-output "$IRIS_OUTPUT_DIR/resolved-config.json" --logging-only
```

After the logging check succeeds, replace `--logging-only` with `--run`.
Pass `--resume-from gs://.../step-N` to continue a verified checkpoint. The
checkpoint step counts completed updates: `step-5187` resumes zero-based
update 5,187; update 8,820 finishes at checkpoint step 8,821. The source training cache is
`gs://marin-us-central2/datakit/hero_tok/v16384_shuf/train`; the source
tokenizer is `gs://marin-us-central2/tokenizers/hero-bpe-v16384/hf-hub-1.21.0`.
These are existing external artifacts, not new tokenization jobs.

The complete-window result used pipeline depth 1; the checkpoint diagnostic
uses depth 0 for synchronized individual-update comparisons.

The commands above are worker commands. With the configured Marin Iris client,
submit them through `iris --cluster marin job run --tpu v4-32 --cpu 16
--memory 96g --disk 64g --region us-central2 --sync-package marin-root
--sync-package marin-core --extra tpu -- <worker command>`. Iris creates four
coscheduled workers and supplies distributed JAX discovery. Pass the shown
environment variables with Iris `-e NAME VALUE` options, and let Iris supply
each worker's `IRIS_OUTPUT_DIR`. Retain the scheduler's default preemption
and retry policy; no wrapper deadline is required.

For a bounded integration check on the original optimizer schedule:

```bash
export BELAY_RUN_ID=YOUR_UNIQUE_DIAGNOSTIC_ID
export BELAY_TPU=v4-32
python -m experiments.grug.belay_tpu.run_diagnostic_phase \
  experiments.grug.belay_tpu.belay_ckpt1_tpu_update_check \
  --output-path gs://YOUR_BUCKET/YOUR_UNIQUE_DIAGNOSTIC_ID \
  --kernel-profile kda-owner-causal-sort-v1 \
  --initial-checkpoint gs://YOUR_BUCKET/VERIFIED_CHECKPOINT/step-5187 \
  --update-steps 32 --profile-steps 10
```

The diagnostic performs 42 unique updates plus one next-update replay. It
checks complete checkpoint state, data continuation, finite loss and compiler
memory, and saves a ten-update profile. Its short timing excludes data loading
and checkpoint I/O; use the native trainer's complete windows for MFU. Run
the same diagnostic with `--logging-only` first to check its online tracker.

`test_*.py` contains operator, gradient, causality and capacity oracles; these
do not perform optimizer updates. Remote CPU interpreter checks can use
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=16` with
`pytest -q -n 0 experiments/grug/belay_tpu/test_*.py`. TPU compilation and
whole-update checks remain necessary before promoting a changed kernel.

The experimental fused-expert branch is retained in the
[research history](https://github.com/WhenWen/astra-d8-research/issues/253),
outside this selected runtime. Its operator speedup did not improve the
whole-update median, so it is not enabled by any profile here.
