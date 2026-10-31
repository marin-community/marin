# Mini840 SkyRL evaluation runtime

This directory repairs the missing eval-only launcher and environment manifest reported in
[Mini840 #10804](https://github.com/marin-community/marin/issues/10804). It consumes the released
TaskRolloutSpecs, uses Pi 0.87.0 for Harbor tasks, and runs SkyRL's `generate` entrypoint without training.

**Full end-to-end validation is pending.** The released 840 unique tasks expand to 1,626 rollouts.
Their 287 agentic tasks require 197 environment source keys. The available eval account reached its
45-snapshot quota. All three available credential roles reject BixBench's published 20 GB disk
requirement because their per-sandbox maximum is 10 GB. A diagnostic smoke that omits BixBench must
not be reported as full Mini840 validation. No complete approved environment map is published yet.

The environment-key and original-trial code is unchanged between runtime
`815c4b2c8197bcf1613b405d3cad144dbe0b9f81` and this directory's base revision. The new launcher uses its
own checked-out revision; record that revision with results. Earlier package and grader checks did
not exercise this full execution path.

## Frozen inputs

- Dataset: `open-athena/Marin-Mini-Eval-840@bae388a44fc0ba9ffa551a4c6347fbe03a9fefd7`.
- Unique tasks: `data/validation.parquet`; expanded rollouts: `data/validation-rollouts.parquet`.
- Smoke model: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`.
- Source recipe: [`launch.py`](launch.py). Total context is 65,536 tokens, maximum output is
  16,384 tokens per turn, and thinking is enabled. Compiled vLLM uses TP2 on two H100s.
- Judge: the repository's pinned
  [131k MiniMax configuration](../campaigns/eval-campaign-09-25-unlabeled/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml).
  The launcher co-hosts it on eight H100s unless an existing endpoint is supplied. Credentials are
  resolved at submission and are never written into launch YAML.

The published package graders and repetitions remain unchanged. A smoke selects the first unchanged
unique task from each benchmark; it does not replace the complete validation data.

## CPU preparation on RNO2A

Existing Iris, CoreWeave, HF, Daytona and Together credentials are required. `DAYTONA_RL_API_KEY`
identifies the account that must own the prepared snapshots. Choose another explicit credential
role if that account lacks the required capacity. Preparation may build images; evaluation cannot.

```bash
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-rno2a --priority batch --cpu 4 --memory 16GB --disk 128GB \
  --timeout 3600 --job-name mini840-stage-inputs \
  -- uv run --with-editable ./vendor/harbor --with daytona==0.200.2 --prerelease allow \
  python -m experiments.evaluation.mini_policy_runtime.stage_inputs
```

The staging report supplies the complete model mirror and data URIs under
`s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031/`. Preparation restores the
original published environment files and computes keys on Linux using the same code as the runtime.
It records snapshot IDs, source keys, package hashes, credential role, dataset revision, and failures.
It stops launching further builds when the snapshot quota is reached. Existing shared snapshots are
not deleted.

Once account limits permit the full cohort:

```bash
uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-rno2a --priority batch --cpu 4 --memory 16GB --disk 32GB \
  --timeout 7200 --job-name mini840-prepare-environments \
  -e DAYTONA_RL_API_KEY "$DAYTONA_RL_API_KEY" \
  -- uv run --with-editable ./vendor/harbor --with daytona==0.200.2 --prerelease allow \
  python -m experiments.evaluation.mini_policy_runtime.prepare_environments \
  --credential-env DAYTONA_RL_API_KEY \
  --data-uri s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031/full
```

Use `--smoke` with the staged `/smoke` data URI for one task per benchmark. Diagnostic benchmark
subsets must be explicit and use a different data URI. A snapshot map for a smoke cannot launch the
full dataset. The launcher's missing-key and dataset-scope checks run before GPU artifact creation.

## Plan and launch

After the CPU reports succeed, use the model URI in the staging report and the complete environment
manifest URI. Pin the published manifest to immutable storage before sharing it with other users.
The example below still fails while the account limits above are unresolved.

```bash
uv run python -m experiments.evaluation.mini_policy_runtime.launch \
  --version 2026.10.31 \
  --model-uri "$MODEL_URI" \
  --data-uri s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031/full \
  --environment-manifest "$APPROVED_MANIFEST_URI"
```

Inspect the plan, then add `--run` to the same command. The coordinator and model jobs run on RNO2A
at batch priority. The GPU task uses the privileged container profile for local bubblewrap graders.
The complete cohort retains all 1,626 released rollout rows. The immutable resolved SkyRL launch
configuration, trajectories and Harbor trial evidence identify the actual run.

A successful E2E check must include real model responses, Pi tool use, original Harbor verification,
answer-package grading, and the judge fallback. Configuration parsing, an HTTP health check, or a
non-empty reward file alone does not establish that result.
