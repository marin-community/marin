# Async non-agentic RL experiments

The [Qwen recipe](async_rl.py), [Snowball recipe](async_snowball.py) and
[terminal auditor](async_rl_audit.py) support controlled Megatron RL experiments
launched through Marin. Both recipes use GRPO advantages and math-verifier
rewards. Run results and decisions are tracked in [epic #8936](https://github.com/marin-community/marin/issues/8936).

## Sources and artifacts

Record the Marin commit, the pinned MarinSkyRL runtime commit, and the emitted
request envelope for every run. A clean launch checkout prevents unrelated local
edits from entering an Iris source bundle. MarinSkyRL must be fetchable at its
pinned Git commit. Preserve the actual execution attempt ID; the dry-run attempt
ID can differ.

The recipes fingerprint model/data dependencies, topology, seed and configuration.
Use one immutable artifact version within a comparison family. Snowball uses the
existing regional SFT export and its thinking template. Keep artifacts and jobs
in the same region; the recipe validates the regional storage prefix.

Check live H100 capacity before launching. Snowball currently allocates 32 learner
GPUs and eight inference GPUs per replica. Each inference replica occupies one
node with data/expert parallel sizes of eight. `--inference-replicas 2` therefore
requests 48 GPUs in total. Physical GPU identities and memory telemetry must
qualify an allocation before its timing is interpreted.

W&B tracking is enabled at every training step. Configure `WANDB_ENTITY` through
the launch environment; account selection is not part of the recipe. Propagate
an explicit override through any nested environment loader. Verify the actual
W&B URL after startup.

## Preview a Snowball development run

The commands and controls below describe Snowball. The Qwen recipe exposes its
own scales and defaults; use its `--help` and explicitly disable KL for the
no-KL screening family. From the Marin checkout, install its environment with
`uv sync`, and configure regional `MARIN_PREFIX` before previewing:

```bash
uv run --no-sync python -m experiments.post_training.async_snowball \
  --version "$RL_STUDY_VERSION" \
  --scale qualification --runner async --completion metrics \
  --study-steps 100 --eval-interval 20 --seed 17 \
  --response-tokens 4096 --eval-response-tokens 4096 --context-tokens 8192 \
  --inference-replicas 1 --weight-sync-interval 2 --max-staleness-steps 1 \
  --correction behavior_clip --epoch-seeded-shuffle \
  --timeout-seconds 7200 --dry-run
```

Set `RL_STUDY_VERSION` to the family's chosen immutable version before invoking
the command. The preview builds the request without launching training. Replace
`--dry-run` with `--run` inside a bounded Iris CPU coordinator to execute it.
For example, after checking capacity and setting `WANDB_ENTITY`, `IRIS_USER` and
the immutable version, submit from the same checkout:

```bash
uv run --no-sync iris --cluster marin job run \
  --target-cluster cw-us-east-02a --priority batch \
  --job-name snowball-development-c2-a1 \
  --cpu 4 --memory 16GB --disk 32GB --enable-extra-resources \
  --max-retries 0 --timeout 7800 \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e MARIN_CLUSTER coreweave -e WANDB_ENTITY "$WANDB_ENTITY" \
  -- python -m experiments.post_training.async_snowball \
  --version "$RL_STUDY_VERSION" \
  --scale qualification --runner async --completion metrics \
  --study-steps 100 --eval-interval 20 --seed 17 \
  --response-tokens 4096 --eval-response-tokens 4096 --context-tokens 8192 \
  --inference-replicas 1 --weight-sync-interval 2 --max-staleness-steps 1 \
  --correction behavior_clip --epoch-seeded-shuffle \
  --timeout-seconds 7200 --run
```

Iris performs the remote workspace dependency setup; do not pass its `--no-sync`
option to a fresh coordinator image. Give the coordinator enough time for
dependency preparation and the training deadline. Record failed attempts alongside
successful attempts in elapsed-resource accounting.

For a five-update correctness gate, use `--scale cadence-gate`, omit
`--study-steps`, and set `--eval-interval 5`. A synchronous comparison uses
`--runner sync --weight-sync-interval 1`. The synchronous runner publishes after
every update; it does not implement delayed publication.

## Experimental controls

| Control | Meaning |
| --- | --- |
| `--weight-sync-interval C` | Publish inference weights every C completed optimizer updates; final evaluation receives a final publication. |
| `--max-staleness-steps A` | Maximum admitted group age in completed updates; requires C ≤ A + 1. Measure realized ages as well. |
| `--correction behavior_clip` | Reference objective using learner/behavior probability ratios and clipping. |
| `--correction regular_tis` | Regular PPO plus truncated importance sampling, cap 2, with required rollout logprobs. This changes an objective package. |
| `--epoch-seeded-shuffle` | Shared seed-plus-epoch source permutation. Async completion can still change consumed order and samples. |
| `--completion metrics` | Durable training result and evaluations; no native checkpoint or HF model export. |
| `--completion checkpoint` | Retain resumable training state. |
| `--completion model` | Retain training state and run a separate HF export stage. This is the default. |

Keep model, tokenizer, prompt/verifier, train rows, sampling, optimizer, batch,
microbatch, token budgets and evaluation schedule fixed within a comparison.
Declare the changed controls before launch. An evaluation between scheduled
publications can force an extra weight update, so the development interval of 20
is divisible by both C2 and C4.

The first 128 GSM8K test questions are reused development data. For confirmation,
freeze configurations and the endpoint first, then use
`--validation-offset 128 --validation-rows 1191 --eval-interval 100` with
`--study-steps 100`. This selects the remaining test questions and evaluates
initial/final weights. Run each selected configuration at seeds 17, 29 and 43.
Do not choose checkpoints or tune configurations using these confirmation
results. The stored validation-window manifest lets the auditor verify the
dataset revision and row range before reading scores.

## Verify a completed run

Run the locked CPU auditor in the artifacts' region:

```bash
uv run --locked --script experiments/post_training/async_rl_audit.py \
  --spec audit-spec.json
```

The script's module docstring defines the specification. Supply actual run and
attempt IDs, receipt URI, request envelope, W&B URL, expected update/evaluation
counts, and expected configuration/provenance. The auditor checks finished W&B,
finite metrics, probability coverage, receipts, evaluation hashes and aggregates,
and successful launcher/Iris completion. `locked_validation` additionally locks
the dataset revision and row window. Training evidence from a failed launcher
can be inspected explicitly, but does not count as clean end-to-end completion.

The artifact's `terminal.json` contains the actual `request`, `execution` and
`response`; extract its first two fields plus `schema_version` as the audit
envelope. `response.training.receipt_uri` identifies the receipt. Failed attempts
have their own JSON under `request.output.attempts_root`. The CLI's fingerprint
preview can contain placeholder dependency locations, so use the executed manifest
for auditing. Obtain the actual W&B URL from the training logs.

Verify Finelog separately. The [Async RL Training dashboard](../../infra/grafana/dashboards/async_rl.json)
selects run, job and execution identities. Check exact optimizer updates,
publication versions and realized ages; reconcile generated tokens with consumed,
discarded and cancelled outcomes; inspect terminal telemetry delivery. A missing
correction-skip series is not proof of zero skipped updates.

## Interpret the measurements

Core RL seconds sum `phase_duration_seconds` with `phase=step` in Finelog;
`async/performance/core_seconds` is the corresponding async step metric. This
excludes startup and evaluation, and includes the final weight publication.
Concurrent rollout-call durations cannot be summed into elapsed time. Report publication, learner work and waiting
alongside useful token/sequence counts; equal update budgets can produce different
token work. Publication transfers weights from learner to inference, not gradients.

Report allocation × core seconds / 3,600 separately from task-running GPU-hours,
which sum every GPU attempt's running duration × its GPU count. The latter includes
setup, evaluation and teardown, but excludes pre-running reservations and is not
a billing measurement. Disjoint GPUs do not exclude shared-fabric contention;
use repeated, counterbalanced execution when confirming timing differences.

Policy drift includes learner/behavior log-ratio statistics, token-weight effective
sample size and clipping pressure. Update age is a provenance measurement, not an
exact probability distance. Backend probability differences can exist at zero age.
Compare fixed-endpoint reward against core time and measured drift, and retain
response length, truncation and discarded-work diagnostics.

Optional `paired_studies` in the audit spec compares matched evaluation question
IDs across configurations and observed training seeds. Its bootstrap resamples
questions jointly across arms and seeds; intervals are conditional on those
training seeds. Report per-seed results and spread as well. A wide interval does
not establish quality equivalence. The initial study may recommend the conservative
configuration while leaving the acceptable drift range unresolved.
