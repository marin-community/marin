# Mini840 SkyRL evaluation runtime

This directory repairs the missing eval-only launcher and environment manifest reported in
[Mini840 #10804](https://github.com/marin-community/marin/issues/10804). It consumes the released
TaskRolloutSpecs, uses Pi 0.87.0 for Harbor tasks, and runs SkyRL's `generate` entrypoint without training.

**Full end-to-end validation is pending.** The released 840 unique tasks expand to 1,626 rollouts.
Their 287 agentic tasks use 197 distinct environment sources. The Daytona eval account can build and
run these sources directly; named snapshots are optional. The earlier mandatory snapshot-map gate
was a runtime defect. CPU preflight checks native environment startup and Pi installation before
allocating model GPUs. A partial preflight is not full Mini840 validation.

BixBench's embedded task manifest requests 20 GB, but its original policy config explicitly overrides
that request to 10 GB. Preparation reads this override from the frozen policy index and records it
by environment source key. The installed-agent runtime applies the same override to the original
Harbor trial. Other tasks retain their own resource budgets, including TBLite's 2 GB task. The older
`815c4b2c8197bcf1613b405d3cad144dbe0b9f81` runtime lacks these source-specific overrides; use the
repair revision when running the new launcher. Earlier package and grader checks did not exercise
this full execution path.

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

The grader code and repetitions remain unchanged. Runtime packaging repairs remove unsupported local
resource allocations, publish wheels for source-only dependencies, and bind OlympiadBench judge
credentials into the verifier sandbox. Repaired data receives a new immutable revision. A smoke selects the first unchanged
unique task from each benchmark; it does not replace the complete validation data.

## CPU preparation on RNO2A

Existing Iris, CoreWeave, HF, Daytona and Together credentials are required. Use the explicit
`DAYTONA_EVAL_API_KEY` role for native Harbor environments. No snapshot quota is needed.

The repair sequence is: build the pinned grading dependency wheels with `repair_release
--build-wheels`, stage the repaired inputs using `stage_inputs --wheels-revision REVISION`, then run
`environment_preflight --credential-env DAYTONA_EVAL_API_KEY --data-uri DATA_URI`. All three commands
run on RNO2A CPU jobs. Add `--smoke` to preflight for the 21-benchmark smoke dataset. The full preflight
checks all 197 environment sources; benchmark-specific diagnostic preflights cannot launch an eval.

Commands and immutable repaired dataset revisions will be recorded here after the live checks pass.

## Plan and launch

After the CPU reports succeed, use the model URI in the staging report and the complete environment
manifest URI. Pin the published manifest to immutable storage before sharing it with other users.
An incomplete manifest fails before GPU artifact construction.

```bash
uv run python -m experiments.evaluation.mini_policy_runtime.launch \
  --version 2026.10.31 \
  --model-uri "$MODEL_URI" \
  --data-uri s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031/full-runtime-v3 \
  --environment-manifest "$APPROVED_MANIFEST_URI"
```

Inspect the plan, then add `--run` to the same command. The coordinator and model jobs run on RNO2A; `--priority production` is available
for relaunches after pre-emption. The GPU task uses the privileged container profile for local bubblewrap graders.
The complete cohort retains all 1,626 released rollout rows. The immutable resolved SkyRL launch
configuration, trajectories and Harbor trial evidence identify the actual run.

A successful E2E check must include real model responses, Pi tool use, original Harbor verification,
answer-package grading, and the judge fallback. Configuration parsing, an HTTP health check, or a
non-empty reward file alone does not establish that result.
