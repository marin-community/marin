# EMA-Gram polar exponent experiment

Compare `alpha = 0, 1/8, 1/4, 1/3, 1/2` using the Delphi 447M recipe,
then conditionally repeat the decisive comparison at 998M. This directory
implements the experiment protocol; it contains no measured training results.
Track execution and findings in [issue #10784](https://github.com/marin-community/marin/issues/10784).

The starting recipes are
[`train_delphi_seeds.py`](../references/train_delphi_seeds.py) and the
[Delphi seed-repeat guide](../../docs/reproducibility/delphi-seed-repeats.md).
The 447M recipe has 37,335 updates of 8 sequences × 4,096 tokens
(1,223,393,280 tokens); the 998M recipe has 38,014 updates of 32 × 4,096
(4,982,571,008 tokens). Pilots use 7,630 updates (250,019,840 tokens).

Prior experiments: [left-preconditioned Muon #6588](https://github.com/marin-community/marin/pull/6588),
[activation-aware Muon #6538](https://github.com/marin-community/marin/pull/6538),
[gain-gated Muon #6506](https://github.com/marin-community/marin/pull/6506), and
[error-aware Muon issue #4919](https://github.com/marin-community/marin/issues/4919).
The final comments on #6588 found that an approximate inverse square root
changed the result substantially. This experiment uses spectral Gram powers
and separately measures the error of the production polar approximation.

## Update and fixed controls

For each flattened Linear weight, including each scanned layer independently:

```text
M <- .95 M + .05 G
L <- beta2 L + (1-beta2) G G^T
R <- beta2 R + (1-beta2) G^T G
A = L / (trace(L)/rows) + ridge I
B = R / (trace(R)/cols) + ridge I
direction = A^-alpha polar(A^-alpha M B^-alpha) B^-alpha
W <- W - lr * (direction + weight_decay * W)
```

Buffers start at zero, and a zero Gram uses identity on that side. Powers use
FP32 `eigh` after the EMA update, with negative roundoff eigenvalues clipped to
zero before adding the ridge. Grams and momentum stay FP32. The production
polar is Levanter's five-step quintic Newton–Schulz approximation; the offline
SVD reference removes singular modes below `eps * max(rows,cols) * sigma_max`.
The negative sign is applied once by the learning-rate transform.

The native trainer finishes batch accumulation and global gradient reduction
before calling the optimizer. A single global norm clip at 0.1 precedes both
parameter groups and the Gram updates. The optimizer does not average
worker-local Grams. Full matrices are used, without blocking, and both Grams
refresh every update, including passively for alpha=0. No Nesterov, bias
correction, aspect-ratio scaling, or hyperball projection is added.

All cells share Muon's Linear-weight routing, with embeddings, output head,
biases and norms on AdamW. Its LR, betas and epsilon come from the corresponding
Delphi recipe and remain fixed during matrix-LR tuning. Weight decay is 0.1
with the same mask across exponents. This replaces the recipe's AdamH optimizer;
it preserves its architecture, tokenizer, cache order, data mixture, batch,
context, precision and linear schedule shape (10% warmup, 20% decay).
Pilot schedules use the pilot horizon; confirmation schedules use the full
recipe horizon. Confirmations start from scratch in distinct artifact paths.
Seeds 0, 42 and 62746 pair model initialization and data order explicitly.

## Plan, tune and lock

Run from the repository root. The five LR centers are required inputs in
exponent order. The values below are illustrative starting grids, not tuned
recommendations; calibrate them before using compute.

```bash
uv run python -m experiments.ema_gram.plan pilot \
  --centers .01 .005 .002 .001 .0005 --output /tmp/ema-pilots.json
uv run python -m experiments.ema_gram.train --plan /tmp/ema-pilots.json \
  --version 2026.10.30 --output /tmp/ema-configs.json
```

The second command only writes all 30 resolved configs. Each nonzero exponent
gets LR multipliers `{0.5, 1, 2}` and ridges `{1e-4, 1e-2}`. Alpha=0 gets six
log-spaced rates with exponents `{-2.5,-1.5,-0.5,0.5,1.5,2.5}` around its center
and one ridge (which has no effect). Cell names include a hash of all cell
settings. Use a new calendar version for an independent campaign.

Before a launch, run the same `train` command with `--logging-check` on the
execution host. This initializes online W&B, logs the resolved config and a
logging-check metric, prints the run URL, and performs no optimizer steps.
Authentication errors must be resolved before training. `--run` submits the
selected plan through Marin's StepRunner; it is separate from the logging check.
The recipe requests v4-8 and 120 CPU cores in us-central2, where its existing
caches live. Host RAM is 64 GiB for 447M and 128 GiB for 998M, fixed across
variants within each model size. Host RAM is a scheduling reservation; TPU HBM,
batch size, model precision and all training controls stay fixed across cells.
No job time limit or preemption-retry override is configured. Default
concurrency is one; inspect capacity and per-step memory/time before a sweep.

Every training process gets a fresh online W&B ID, even after preemption.
The stable trainer ID and checkpoint path let the native trainer restore
complete state. A process restart alone does not establish checkpoint resume;
inspect its restore logs. Incremental JSON metric logs and peak-memory snapshots
are written locally to `logs/<trainer-id>/<attempt>/`. Immutable metric chunks
are also persisted every 100 records under `<output-path>/attempts/<attempt>/`,
with the remaining tail and memory snapshot saved on normal exit. Abrupt worker
loss can lose the last partial chunk; online W&B and task logs provide additional
evidence. Upload errors fail visibly.
W&B runs share `marin-community/marin` and group `ema-gram-<version>`.

Create a JSON run manifest containing one selected attempt per cell:

```json
[{"cell": {"alpha": 0.25, "ridge": 0.0001, "learning_rate": 0.002,
           "seed": 0, "phase": "pilot", "slug": "3e18", "horizon": 100},
  "run_path": "marin-community/marin/ACTUAL_RUN_ID"}]
```

```bash
uv run python -m experiments.ema_gram.report --runs /tmp/ema-runs.json \
  --output /tmp/ema-results
uv run python -m experiments.ema_gram.plan expand --plan /tmp/ema-pilots.json \
  --results /tmp/ema-results/results.json --output /tmp/ema-expanded.json
```

Expansion adds one factor-of-two point beyond every edge-winning LR grid.
Run new cells, retain the old results, and repeat until all winners are interior.
The lock command rejects missing, unfinished, short, nonfinite, duplicate or
edge-winning results. Explicitly failed cells remain in the record and cannot
win. Selection uses final `eval/paloma/c4_en/loss` in nats.

Choose a common held-out loss target before confirmation. For example, replace
`TARGET` below with that numeric target after inspecting pilot curves:

```bash
uv run python -m experiments.ema_gram.plan lock --plan /tmp/ema-expanded.json \
  --results /tmp/ema-results/results.json --loss-target TARGET --output /tmp/ema-lock.json
uv run python -m experiments.ema_gram.plan confirmation \
  --lock /tmp/ema-lock.json --output /tmp/ema-confirm.json
uv run python -m experiments.ema_gram.train --plan /tmp/ema-confirm.json \
  --lock /tmp/ema-lock.json --version 2026.10.30 --output /tmp/ema-confirm-configs.json
```

The lock records pilot evidence, tuned LR/ridge, target and a digest. It is
revalidated before generating three paired seeds for alpha=0, 1/4 and 1/2,
plus 1/8 or 1/3 if it is the best pilot. `train` also checks confirmation plans
against the lock. Retain the lock before any confirmation starts.

## Checkpoint probes and cost measurements

Permanent full-state checkpoints are retained every `floor(steps/10)` updates.
Choose the checkpoints nearest 10%, 50% and 90% of one reference confirmation;
weights-only HF exports are insufficient. For the 447M reference these are
updates 3,733, 18,665 and 33,597. Use the actual checkpoint directories from
its logs, since checkpoint filenames use the trainer's step convention.

```bash
uv run python -m experiments.ema_gram.probe \
  --configs /tmp/ema-confirm-configs.json --config-index 0 \
  --checkpoint /path/to/full/checkpoint \
  --dataset uncheatable_eval/wikipedia_english --batches 8 \
  --step-sizes 1e-5 3e-5 1e-4 --output /tmp/ema-probe.json
```

The probe requires all model and optimizer leaves. It does not advance momentum
or either Gram. Each matrix's direction is rescaled to that matrix's alpha=0
Frobenius norm; nonmatrix weights stay fixed and decay is omitted from this
local direction test. Every exponent/ridge/step uses the same independent
held-out batches. The output retains all batch losses and paired delta mean,
standard deviation and standard error. The default probe corpus differs from
the pilot-selection corpus. Batch offsets allow additional disjoint probes.

One attention and one MLP matrix from the checkpoint also receive offline SVD
comparisons at every exponent/ridge. Outputs include cosine, relative error,
and synchronized warm timings for production directions, SVD directions and
Gram powers. Memory statistics are recorded when the backend supplies them.
These isolated timings exclude distributed communication and training overlap.

Training captures an XProf trace at steps 100–109 for every cell. Named
`update_gram_state`, `relative_gram_power` and `gram_direction` scopes separate
optimizer work; inspect collective operations in the trace for communication
cost and overlap. Report optimizer and decomposition time as well as exposed
communication time, using matched alpha=0 traces. Preserve profiles from their
30-day temporary upload location if needed longer. Do not substitute isolated
matrix timings for full training overhead.

## Confirmation reporting and conditional follow-ups

```bash
uv run --package marin-levanter --extra dev python -m experiments.ema_gram.report \
  --runs /tmp/ema-confirm-runs.json --lock /tmp/ema-lock.json \
  --output /tmp/ema-confirm-results --plot
```

The report retains unsampled W&B histories, configurations, summaries,
individual seed curves, exact-token paired differences, and first observed
crossings of the locked loss target. Crossings are discrete evaluation points,
not interpolated estimates. Failed and incomplete attempts remain visible.
Confirmations pair against alpha=0; horizon and scale comparisons pair alpha=1/2
against alpha=1/4 at each seed and horizon. Missing completed references are errors.
Reports verify the logged model, optimizer, data and training/evaluation controls
before assigning token counts. Figures show loss versus tokens and elapsed time, and paired differences;
no smoothing or seed aggregation is applied. A restarted attempt's elapsed
time is excluded from full-run comparisons. Earlier attempts must be exported
separately and joined only after checkpoint restoration and elapsed-time
provenance have been verified; this tool does not infer that lineage.

After the confirmation result is judged reproducible, `plan horizon --lock ...`
generates the 1/4-versus-1/2 comparison with horizons 100, 1,000 and 10,000
updates, and `plan scale --lock ...` generates the 998M comparison. Both use
three paired seeds; `train` requires `--reproducible-result <reviewed-report>`
for these follow-ups. At 447M the horizons are 3,276,800, 32,768,000 and
327,680,000 tokens. The 998M horizon of 100 updates is 13,107,200 tokens.
The report is a human scientific gate, not an automated significance claim.

Publish results with tuned settings, all seed curves and paired differences,
checkpoint probe variation, stability failures, peak memory and cost profiles.
A conclusion about the best exponent requires those measurements; local
numerical tests establish implementation behavior only.
