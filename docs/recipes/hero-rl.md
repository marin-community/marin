# Hero RL qualification at 4K

`experiments.post_training.hero_rl.launch` binds one completed Hero HF export to a
fixed-weight numerical probe and a separate four-update asynchronous RL trial.
Both use the maintained Marin artifact launcher and vendored SkyRL Megatron/vLLM
runtime. Planning builds the graph locally; `--run` submits the selected stage.

## Isolated ShortConv PIECEWISE runtime experiment

This source snapshot stages an experimental overlay into every GB200 task's
private vLLM environment. It is intended only for the GB200 experiment below.
The canonical runtime and its production dependency pins remain separate.
Frozen dependency installation is followed immediately by
`experiments/post_training/hero_rl/piecewise_runtime.py`, before runtime import
checks. The stager verifies the exact 39e ARM wheel receipt, dependency versions,
three source preimages and fixed payload hashes. uv 0.13.5 may record an empty
`archive_info` in `direct_url.json`; the stager independently downloads the exact
locked wheel, requires its SHA256, and compares every installed vLLM payload
file with that verified ZIP, including the complete native-library content set.
The streaming download has a 4 GiB archive limit, a 15-minute deadline and a
120-second socket timeout. The archive remains in disposable task storage only
during staging. The stager copies the entire package
with symlinks dereferenced, verifies unchanged native extensions, then publishes
the three Python changes together by replacing the environment's package entry.
The shared uv package cache remains unchanged.

The overlay manifest SHA256 is
`266cb34f775746ac050b13f4dcce38a615ae1718b450c546fba4ca4c06dfca8d`.
The source patch SHA256 is
`1b377716b4586cc8ad5e4c607048aa6bb39ce159893da847cf6514ba9aae1896`
against vLLM `39e62869693c46402b1a95fde4fc55ca1aab9ae1`.
GB200 artifact names include `piecewise-266cb34f775746ac`, and engine settings
request `PIECEWISE` with capture sizes `[1, 2, 4, 8, 16]`. Batch invariance and
routed-expert capture remain enabled. Full CUDA graphs remain unsupported.

The CPU tests establish staging and launch composition only. Keep the
qualification hold until the seven kernel/config cases, a matched route-enabled
DP4 engine comparison, and fresh full-Hero numerical qualification all pass.
Do not submit the update stage before those gates pass.

Prepare the fixed-weight stage locally with the completed SFT64 export:

```bash
uv run python -m experiments.post_training.hero_rl.launch \
  --version 2026.11.04.31 --hardware gb200-48 --priority interactive \
  --stage numerics --probe-prompts 128 \
  --model-uri s3://marin-us-east-02a/marin/users/dlwh/hero-rl/sft64-bf16-fp32-router/2026.11.03.49 \
  --model-version 2026.10.22.64 \
  --export-manifest-sha256 d6bfdb32b7f061ae6abc6af7c289522f3103ab1b9b772110aa451c3b757a6f25
```

This command does not submit work. Use the reviewed immutable runtime source
commit and a fresh artifact version for any later authorized submission.
Each of the twelve task output directories must contain
`hero-piecewise-runtime.json`; bootstrap also logs that receipt. Require twelve
receipts with this manifest digest, identical source preimages, the verified
wheel archive SHA256, matching installed payload hashes, exact core dependency
identities, and equal before/after extension hashes. Archive
them with the resolved launch and numerical report.

## Inputs and placement

The default initializer is the completed September step-144000 BF16 split-expert
export recorded in [Hero readiness #10388](https://github.com/marin-community/marin/issues/10388):

```text
s3://marin-us-east-02a/marin/users/romain/hero-vllm-rl/hero-535b-step144000-bf16-split-v1
```

Its `export-manifest.json` SHA256 is
`b20fdd303eb6ff8993ae5b5d7add14d65787e70981c056fea6de9dea62acfae1`.
A CPU prerequisite checks this digest, the Hero architecture and the HF index
before GPU allocation. It also exercises the runtime's model-metadata staging
and loads the separately pinned tokenizer in SkyRL's CPU environment. The model
export does not need bundled tokenizer files. This check does not download or
rehash model shards.
The model is adopted in place. The graph has no initializer-export producer,
latest-checkpoint lookup or checkpoint conversion step.

To qualify a different completed export, pass all three input options:
`--model-uri`, `--model-version` (an immutable calendar version) and
`--export-manifest-sha256`. A native checkpoint or an incomplete export is not an
HF initializer. Retain its config, index, manifest and source lineage together.
The default export remains at its existing CoreWeave object-store URI for either
hardware profile. The GB200 profile reads that URI directly, as the historical
GB200 trial did; the graph creates no durable copy of its approximately 1.07 TB
of weights. A different completed export must have its own pinned identity.

The tokenizer is `marin-community/marin-tokenizer` at
`a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`. Its pinned chat template is used by
the runtime, and EOT token 128009 stops responses. GSM8K comes from the existing
`Gsm8kParquetConfig` writer at pinned revision `e53f048`: the first 1,024 train
rows and first 128 test rows, with the explicit `#### number` answer contract.
This is a newly rendered qualification cohort; it does not reproduce the
historical run's exact dataset bytes.

The default `--hardware h100` requests 19 eight-GPU H100 nodes in `cw-us-east-02a`: 128 learner
GPUs at TP1/PP16/CP2/EP8/ETP1 and 24 separate serving GPUs at TP1/PP1/DP24/EP24.
The typed role plan selects multiprocessing placement so the serving engine can
span three nodes. Publication uses broadcast/NCCL, captured-expert replay is
enabled, and query-balancing bias remains frozen. Training uses the historical
MuonH/AdamH/Adam optimizer route at `1e-6` with CPU momentum offload.

The request cap is 4,096 tokens, split into at most 1,024 prompt tokens and 3,072
generated tokens. The historical trial used a 256/3,840 split. Megatron's physical
recompute sequence size remains 4,224; it does not enlarge the served context.
Compilation and CUDA graphs remain enabled. There are no automatic retries;
each GPU job has a six-hour deadline and its coordinator has an eight-hour deadline.

Streaming uses a 16 GiB CPU buffer per serving rank and four loader threads.
RunAI's [unlimited distributed-streaming default](https://github.com/dsx-ai-factory/model-streamer/blob/0.16.1/docs/src/usage.md#cpu-memory-capping)
can retain roughly the full
model payload collectively on each serving tray, exceeding its host-memory
budget. Four ranks per GB200 tray instead have a combined 64 GiB streamer-buffer
cap; this does not include other process allocations. Distributed diagnostics
record per-node cgroup memory during startup. Ray advertises its cgroup budget
minus its bounded object store as heap capacity; that scheduler value is not an
actor RSS limit.

## Experimental 48-GB200 profile

`--hardware gb200-48` requests twelve four-GPU trays in `cw-us-east-08a`:
36 learner GPUs at TP1/PP12/CP1/EP3/ETP1 and 12 separate serving GPUs at
TP1/PP1/DP12/EP12. Each update consumes 15 prompt groups of eight responses,
so its 120-response batch divides evenly across the learner's three dense-DP ranks.
Microbatches remain one. Each tray requests 128 CPUs, 940 GiB host memory and
4 TB disk; serving uses GPU memory utilization 0.90. Full uniform recomputation
also saves activation inputs to pinned CPU memory through
`trainer.policy.megatron_config.transformer_config_kwargs.grug_checkpoint_cpu_offload=true`.

This is an unqualified memory and numerical experiment. The routed-expert BF16
weights, FP32 masters and FP32 gradients alone require 135 GiB per learner GPU.
Dense parameters and optimizer moments raise the estimate to approximately
158 GiB on an embedding or output-head stage, before activations, communication
and kernel workspace. Actual peak memory remains to be measured. CPU momentum
and activation offload also consume host memory.

CP1/EP3 pools three different sequences in each expert group. Historical
GB200 qualification used CP4/EP4, and the
[later numerical investigation](https://github.com/marin-community/marin/issues/10388#issuecomment-6379516756)
found sensitivity to peer-sequence batching. Require a fresh numerical PASS on
this exact hardware profile before submitting its update stage. A pass from the
H100 profile or the historical 48-learner GB200 layout does not qualify it.

Prepare its numerical stage at interactive priority:

```bash
uv run python -m experiments.post_training.hero_rl.launch \
  --version 2026.11.03.48 --hardware gb200-48 --priority interactive --stage numerics --probe-prompts 32
```

The GPU artifact names include `hero-rl/gb200-48/`, so this profile has separate
numerical and update identities. Its mismatch archive is the `mismatch_probe`
child of `checkpoints/hero-rl/gb200-48/numerics-4k/<version>` under the user's
artifact prefix. Priority is an execution choice; `--priority` defaults to
`batch` for existing callers. Pass `--priority interactive` for the validation runs.
Inspect the resolved allocation and placement before adding `--run`.

## Prepare the numerical stage

Build and inspect the graph from the repository root:

```bash
uv run python -m experiments.post_training.hero_rl.launch \
  --version 2026.11.03.1 --priority interactive --stage numerics
```

The numerical stage is synchronous, admits zero staleness and performs zero
optimizer updates. It samples 16 validation prompts with eight independently
seeded responses each, then archives generation-time scores, exact tokens,
captured routes and trainer scores using the existing mismatch-probe callback.
Its archive is the `mismatch_probe` child of that stage's artifact directory.
With the cluster's default `MARIN_PREFIX=s3://marin-us-east-02a/marin`, the
directory is
`s3://marin-us-east-02a/marin/users/<user>/checkpoints/hero-rl/numerics-4k/<version>/mismatch_probe`.
Use the submitting Iris username and the `--version` value. If the coordinator
overrides `MARIN_PREFIX`, substitute that prefix. The stage's adjacent
`resolved-launch.yaml` records `artifacts.resolved_config_uri`; replace its
filename with `mismatch_probe` to recover the archive location.
Route records are split across bounded FineStore transactions, with the
completion manifest published after the data commits.
The update stage is a separate graph terminal, so selecting `numerics` does not
also launch training. After approving the displayed allocation and inputs, add
`--run` to the same command to submit it.

Evaluate a completed archive locally:

```bash
uv run python -m experiments.post_training.hero_rl.report \
  <numerics-artifact-directory>/mismatch_probe \
  --expected-model-source s3://marin-us-east-02a/marin/users/romain/hero-vllm-rl/hero-535b-step144000-bf16-split-v1 \
  --expected-starting-step 0 \
  --output-dir /tmp/hero-4k-numerics
```

The JSON and Markdown reports apply the historical empirical gate: at least
50,000 independent, loss-bearing response tokens, with fewer than 0.1% of
same-token trainer/sampled-serving probability ratios outside `[0.8, 1.2]`.
The report checks source, starting step, exact token alignment, context limit
and captured/executed route agreement across all 48 layers and eight expert choices. It counts initial sampling once and
ignores repeated and later scorings. Exit codes are PASS=0, FAIL=1 and INVALID=2.
If the fixed cohort produces fewer than 50,000 qualified tokens, it has not
passed. Increase the cohort with `--probe-prompts 32` and a fresh version, such
as `--version 2026.11.03.2`. The option accepts at most the 128 validation prompts;
each prompt still has eight responses. Rescoring repetitions do not add samples.

## Prepare the update stage

Once the numerical stage passes, inspect the separate update graph:

```bash
uv run python -m experiments.post_training.hero_rl.launch \
  --version 2026.11.03.1 --priority interactive --stage updates
```

For the GB200 experiment, retain `--hardware gb200-48`, its version and
`--priority interactive` when selecting `--stage updates`, after its fresh
numerical gate passes.

The H100 profile performs four async updates, each with 16 prompt groups and eight responses
per group, at most one update of policy staleness and 32 in-flight groups.
The GB200 profile uses 15 prompt groups per update with the same sample count
and staleness limit.
Baseline and terminal evaluation use the 128-row validation panel. Both stages
start from the explicit initializer with `resume_mode=none`.

Require four actual, non-skipped optimizer updates; finite loss, reward and
gradient norm; captured-route execution; matching published policy versions;
and actual PPO-clipped token fraction below 1% on each normal update. Preserve
the resolved launch, metrics, trajectories and job identities. A successful job
exit or the fixed-weight numerical PASS does not establish these live gates.
Four updates provide an integration check; an independent learning-quality
comparison needs its own longer, fixed-budget experiment.

In W&B project `marin-community/marin-hero-rl-qualification`, select the run
whose name matches `iris.job_name` in `resolved-launch.yaml`. Inspect steps 1–4:

| Metric | Required observation |
| --- | --- |
| `policy/policy_update_steps` | One applied update per step, four total |
| `policy/skipped_steps` | Zero at every step |
| `policy/policy_loss`, `policy/raw_grad_norm`, `reward/avg_raw_reward` | Present and finite at every step |
| `policy/ppo_clip_ratio` | Below 0.01 at every step; this is actual clipping, not ratio pressure |
| `policy/router_replay/hit_fraction` | 1.0 at every step |

For publication, inspect the trainer job's console log for
`Policy weights updated: step=... model_version=... reason=training_step` at
versions 1–4. `resolved-launch.yaml` names the run and `terminal.json` retains
its `iris_job_id`. These receipts and the subsequent rollouts must agree on
policy version; missing evidence leaves this gate unverified.

Checkpoint and HF saving are disabled, including terminal HF export. This keeps
the qualification from writing a roughly five-terabyte optimizer checkpoint or
starting another full-model export. Optimizer restore is therefore a separate
qualification, to be planned explicitly when checkpoint work is wanted. The
recipe does not depend on or change any already-running export.

## Local checks

```bash
uv run pytest tests/rl/test_skyrl.py experiments/post_training/hero_rl -q
```

The launcher test composes both complete documents through the independently
locked CPU SkyRL runtime without submitting jobs. It checks multi-node role
allocation, synchronous versus asynchronous execution, the 4K cap and absence
of checkpoint/export callbacks. Report tests write real local mismatch
archives and exercise token coverage, outlier thresholds and invalid evidence.
These CPU checks prepare the qualification; accelerator parity and learning
remain unmeasured on this recipe until its stages are run.
