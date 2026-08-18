---
topic: iceball-micro
issue: https://github.com/marin-community/marin/issues/7797
description: Package MarinSkyRL as a versioned Marin stage and validate pretrain-to-evaluation composition.
author: rjpower
---

# iceball-micro: Task Logbook

## Scope

- Goal: Run a Qwen3-0.6B-size random-init model through pretraining, SFT, MarinSkyRL RLVR, Evalchemy, and Harbor from one Marin experiment.
- Primary metrics: terminal success and durable artifact for each stage; RL optimizer updates and export; Evalchemy and Harbor terminal records and scores.
- Constraints: `cw-us-east-08a`, interactive priority, no more than 64 GPUs; external runtimes remain outside Marin's root lock; Iris is an execution backend rather than part of the experiment contract.
- Coordinating issue: [#7797](https://github.com/marin-community/marin/issues/7797), under [#7098](https://github.com/marin-community/marin/issues/7098).
- Experiment prefix: `ICEBALL-MICRO`.
- Shared tags: `iceball-micro`, `7797`, `post-training-e2e`.

## Current TL;DR

- Design revision 2: [Weaver artifact](https://loom.rjp.io/s/rkdrs7y2/artifacts/design).
- Independent review: [Weaver artifact](https://loom.rjp.io/s/3axuycpe/artifacts/review), complete in session `64sglqtz`.
- The launch-only package and Marin graph are implemented. Live run `/power/iceball-micro-e2e-20260802` has materialized and tokenized its bounded inputs on `cw-us-east-08a`; its coordinator will be refreshed with the reviewed nested-job-name fix before GPU allocation.

## Decision Log

- 2026-08-01: Marin owns the graph, artifact identity, and backend selection. MarinSkyRL owns RL config translation, allocation adapters, Ray/SkyRL execution, and checkpoint semantics. Evidence: design revision 1.
- 2026-08-01: The Iris adapter supplies an allocated gang to `cloud.iris.run_rl`; calling `run_rl` directly from Marin would skip allocation. Evidence: design revision 1.
- 2026-08-01: The task image digest, launch package commit, and embedded `skyrl-train` commit are logical identity. Cluster/priority/retry and equivalent placement remain runtime-only. Evidence: independent review and design revision 2.
- 2026-08-01: Add a separate launch-only distribution; do not repurpose the empty root package or infer a source checkout. Evidence: independent review and design revision 2.
- 2026-08-01: Marin owns immutable input locators; MarinSkyRL materializes and verifies them on every allocated node. Evaluation consumes only a validated exact HF policy export. Evidence: independent review and design revision 2.

## Hypothesis Queue

### Active

- `ICEBALL-MICRO-001`: A lightweight packaged MarinSkyRL launcher can reproduce the current Iris CLI without adding `skyrl-train` to Marin's lock. Next test: install the pinned launcher in an isolated environment and compare normalized dry-run output.
- `ICEBALL-MICRO-002`: A stable artifact run ID plus per-attempt IDs can make failed SkyRL launches resume safely into one output root. Next test: failure/retry behavior test followed by one forced live retry if a natural preemption does not occur.
- `ICEBALL-MICRO-003`: One 4×GB200 node can run the bounded Qwen3-0.6B policy and rollout topology. Next test: MarinSkyRL dry-run resource plan and an image import probe.
- `ICEBALL-MICRO-004`: A produced `SkyRLModel` can feed the existing unified Evalchemy and Harbor launcher without a model-registry edit. Next test: dependency-resolved evaluation dry run with both mechanisms.

### Blocked

- None.

### Falsified / Dead End

- Calling `cloud.iris.run_rl` as the Marin launch entry point: it assumes an externally managed Ray allocation and cannot allocate the Iris gang. Evidence: design revision 1.
- Adding `skyrl-train` to Marin's root environment: it conflicts with the isolated external-runtime pattern and needlessly imports the CUDA stack. Evidence: design revision 1.

### Promoted

- None.

## Background Research Brief

- Effort: high
- Stop rule: new code and issue searches stopped changing the ownership boundary and critical path.
- Date: 2026-08-01
- Full evidence map and source ledger: [design revision 2](https://loom.rjp.io/s/rkdrs7y2/artifacts/design).

## Entry Log

### 2026-08-01 22:45 UTC - Initial design and peer-review handoff

- Hypothesis: Marin can reproduce the Evalchemy/Harbor external-runtime pattern for SkyRL while keeping the GPU runtime in its immutable image.
- Commit Hash: `e096eccf7` (baseline; no implementation commit yet).
- Commands:
  - `git clone --filter=blob:none --depth 1 https://github.com/marin-community/MarinSkyRL.git <temporary-dir>`
  - `uv run infra/echo/cli.py search "Marin SkyRL integration post-training from Marin evalchemy harbor" --limit 20`
  - `gh issue view 7098 --repo marin-community/marin --json number,title,state,url,labels,body,comments`
  - `gh issue view 7797 --repo marin-community/marin --json number,title,state,url,labels,body,comments`
  - `weaver artifact write design /tmp/iceball-micro-design.md`
  - `loom session launch ... --name iceball-design-review ...`
- Config: Marin `e096eccf7`; MarinSkyRL `cc8c8e8de2e7242d7e18f0563933fea0a26ac649`; proposed live target `cw-us-east-08a`, interactive priority, at most 64 GPUs.
- Result: published [design revision 1](https://loom.rjp.io/s/rkdrs7y2/artifacts/design) and launched independent Codex review session `64sglqtz`.
- Interpretation: the main gaps are a real lightweight launcher package, typed/provenance-bearing RL output, object-store model staging, and a unified produced-model evaluation adapter.
- Next action: reconcile peer-review findings, revise the design, publish ordered work items, then implement.

### 2026-08-01 23:00 UTC - Independent review reconciled

- Hypothesis: the initial ownership split can become implementation-ready by making binary identity, durable retry, package resources, and exact model handoff explicit.
- Commit Hash: `e096eccf7` (baseline; no implementation commit yet).
- Commands:
  - `loom session wait 64sglqtz --timeout 45 --interval 3`
  - `weaver artifact show review`
  - `weaver artifact write design /tmp/iceball-micro-design.md`
- Config: review compared Marin `e096eccf7` with MarinSkyRL `cc8c8e8de2e7242d7e18f0563933fea0a26ac649`.
- Result: the reviewer published [four P0 findings](https://loom.rjp.io/s/3axuycpe/artifacts/review). Revision 2 incorporates all four and is published at the stable [design URL](https://loom.rjp.io/s/rkdrs7y2/artifacts/design).
- Interpretation: implementation must begin upstream in MarinSkyRL. A Marin-only subprocess wrapper would preserve the repository-workspace assumption and cannot prove the trainer/image identity required by #7797.
- Next action: create the launch-only distribution and typed terminal protocol in MarinSkyRL, then pin and consume it from Marin.

### 2026-08-02 01:15 UTC - Packaged launcher and Marin graph implemented

- Hypothesis: a launch-only MarinSkyRL wheel plus a typed Marin artifact adapter can preserve exact runtime identity without adding the training runtime to Marin's environment.
- Commit Hash: Marin baseline `e096eccf7` with uncommitted implementation; MarinSkyRL `a20541cce52e8e3564b14dee9129c0668d705e50`.
- Commands:
  - `uv run pytest cloud/iris/tests/test_artifact_protocol.py cloud/iris/tests/test_launch_defaults.py cloud/iris/tests/test_run_rl_ingress_env.py`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run config/update-external.py --check MarinSkyRL`
  - `uv run python -m experiments.post_training.iceball_micro --version 2026.08.01`
  - `uv run pytest tests/post_training/test_iceball_micro.py tests/rl/test_skyrl.py tests/evaluation/test_evaluation.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: launch package, task image, and embedded trainer are exact logical identity; attempts share one checkpoint root and publish separate manifests; `iceball-micro` uses 4xGB200 for pretraining, SFT, and RL, then 1xGB200 for the unified Evalchemy and Harbor evaluation.
- Result: [MarinSkyRL PR #275](https://github.com/marin-community/MarinSkyRL/pull/275) publishes the independent `marinskyrl-launcher` distribution and terminal protocol. Its focused suite passes 38 tests, including the machine-only stdout contract. Marin's focused suite passes 18 tests, external-pin generation is stable, and the required changed-files lint is green. The resolved graph spans streamed FineWeb-Edu pretraining, No Robots SFT, GSM8K GRPO, and one produced-policy evaluation group.
- Interpretation: the peer-review P0s are enforced at both boundaries: runtime identity is checked before launch, the launcher no longer requires a source checkout, checkpoint retry state is durable, and evaluation accepts only a validated exact HF export.
- Next action: complete branch review and PR publication, then execute the graph on `cw-us-east-08a` and force one checkpoint-backed RL retry.

### 2026-08-02 00:10 UTC - Review findings resolved and launch gate passed

- Hypothesis: the full graph is stable under Marin's generated-pin, test, typecheck, and agentic-review gates.
- Commit Hash: Marin implementation commit containing this entry; MarinSkyRL `17d41b6e2849a4b8a78cdb79f3d0d0db9992c9c0`.
- Commands:
  - `gh pr checks 275 --repo marin-community/MarinSkyRL`
  - `./infra/pre-commit.py --review`
  - `uv run config/update-external.py --check MarinSkyRL`
  - `uv run python -m experiments.post_training.iceball_micro --version 2026.08.02`
  - `uv run pytest tests/post_training/test_iceball_micro.py tests/rl/test_skyrl.py tests/evaluation/test_evaluation.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: the terminal policy export is also the evaluation tokenizer source; programmatic eval selection shares the CLI's suite and unknown-key validation; artifact-launched child jobs reuse `IRIS_CONTROLLER_URL` while laptop CLI runs retain kubeconfig discovery.
- Result: every MarinSkyRL CI check passes; its focused suite passes 39 tests. Marin's focused suite passes 20 tests, the exact `17d41b6` pin regenerates cleanly, and Ruff, Black, Pyrefly, structural checks, and branch review pass. The two review findings were resolved by naming the parsed Git source fields and centralizing evaluation selection.
- Interpretation: the graph is now ready for live execution. A final protocol audit also found and fixed stdout mixing before launch; `marinskyrl` now reserves stdout for its single terminal JSON response and streams human logs to stderr.
- Next action: publish the Marin implementation PR, submit the coordinator job, and monitor every stage to terminal evidence.

### 2026-08-02 00:27 UTC - Live graph started after coordinator dependency recovery

- Hypothesis: one CPU coordinator can drive the full graph and reuse the cluster's ambient controller for the nested SkyRL job.
- Commit Hash: Marin `2d8864273`; MarinSkyRL `17d41b6e2849a4b8a78cdb79f3d0d0db9992c9c0`.
- Commands:
  - `uv run iris --cluster=cw-us-east-08a job run --job-name iceball-micro-e2e-20260802 ... -- python -m experiments.post_training.iceball_micro --version 2026.08.02 --run --max-concurrent 4`
  - `uv run iris --cluster=cw-us-east-08a job summary /power/iceball-micro-e2e-20260802`
  - `uv run iris --cluster=cw-us-east-08a job stop /power/iceball-micro-e2e-20260802`
- Config: interactive priority; CPU parent with `marin-core:cpu` and `marin-core:dedup`; child stages request at most 4xGB200, and evaluation requests 1xGB200.
- Result: the first parent attempt failed before GPU allocation because package-scoped sync omitted `marin-dupekit`, an import-time dependency of tokenization. The parent was stopped and resubmitted with `marin-core:dedup`. The corrected attempt passed imports and began the No Robots, FineWeb-Edu, and GSM8K input stages under the same artifact version.
- Interpretation: this is coordinator environment recovery, not an experiment fork. The first live failure identified an exact one-off CLI requirement that belongs in the workflow documentation.
- Next action: monitor data completion, pretraining, SFT, RL checkpoint/retry, and both evaluation records.

### 2026-08-02 00:52 UTC - Input artifacts validated and nested job naming corrected

- Hypothesis: completed CPU artifacts can survive a coordinator refresh, allowing a reviewed launcher-boundary fix before any GPU work is spent.
- Commit Hash: live Marin bundle `2d8864273`; live MarinSkyRL `17d41b6e2849a4b8a78cdb79f3d0d0db9992c9c0`; rebased MarinSkyRL PR head `086a4e248b4ec5bd9f09a28c37cb8d659034ad54`.
- Commands:
  - `uv run iris --cluster=cw-us-east-08a query "SELECT job_id,state,error FROM jobs WHERE job_id LIKE '/power/iceball-micro-e2e-20260802%' ..." -f csv`
  - `gh api repos/marin-community/marin/pulls/7883/comments --paginate ...`
  - `uv run pytest tests/rl/test_skyrl.py`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: FineWeb-Edu uses 4,096 source rows and Qwen3 tokenization; nested SkyRL jobs derive their Iris name from the logical run and attempt IDs after replacing Iris-invalid characters.
- Result: data materialization produced 4,096 FineWeb-Edu rows, 9,500 No Robots rows, and 1,024/128 GSM8K train/validation rows. Tokenization produced 4,268,269 tokens from 4,096 documents and the cache probe read the committed shard ledger. GitHub review found that the artifact name `checkpoints/iceball-micro-rl` would place `/` in the nested Iris `job_name`; the boundary now emits `checkpoints-iceball-micro-rl-2026.08.02-<attempt>` and a behavior test reads the actual launcher request. Seven RL adapter tests and the required lint/typecheck gate pass. The rebased MarinSkyRL PR passes all six CI jobs and Marin now pins its exact `086a4e2` head.
- Interpretation: restarting only the CPU coordinator before pretraining is cheaper and safer than allowing a known deterministic failure after SFT. Durable artifact identities make this recovery reuse the completed data and tokenization outputs.
- Next action: commit and push the fix, refresh the coordinator bundle under the same experiment version, and continue through the GPU stages.

### 2026-08-02 01:10 UTC - Pretraining recovered and entered the train loop

- Hypothesis: the immutable input artifacts and checkpoint root allow coordinator-only recovery without changing the experiment identity or rematerializing data.
- Commit Hash: live Marin bundle `49fabbb952043dbd41e4d2ab5cfe7c7df00f52bf`; live MarinSkyRL launcher `086a4e248b4ec5bd9f09a28c37cb8d659034ad54`.
- Commands:
  - `uv run iris --cluster=cw-us-east-08a job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-ed7cbe30`
  - `uv run iris --cluster=cw-us-east-08a job logs /power/iceball-micro-e2e-20260802/run_levanter_train_lm-ed7cbe30 --since-seconds 300`
- Config: one 4xGB200 pretraining worker; Qwen3 595,769,344-parameter architecture; sequence length 512; global batch 32; 16 optimizer steps; W&B entity `marin-community`, project `marin-iceball-micro`, run ID `iceball-micro-pretrain`.
- Result: the corrected coordinator reused every completed data and tokenization artifact. One earlier GPU child reached W&B setup but failed before optimizer step 0 because the submitted entity `dogml` returned HTTP 403. The coordinator was stopped and resubmitted with the repository-standard `marin-community` entity. The replacement child connected to [W&B](https://wandb.ai/marin-community/marin-iceball-micro/runs/iceball-micro-pretrain), loaded the committed 4,268,269-token cache, initialized from random weights, and entered its first compiled training step.
- Interpretation: all three coordinator recoveries happened before durable optimizer progress and reused the same artifact version. They exposed two required launch-environment inputs (`marin-core:dedup` and a writable W&B entity), while the reviewed nested-name fix was deployed before RL allocation.
- Next action: capture terminal pretraining metrics and export, then monitor SFT and the packaged nested SkyRL launch.

### 2026-08-02 01:52 UTC - Pretraining completed; SFT cache coordinator OOM diagnosed

- Hypothesis: the SFT failure is caused by Zephyr's undersized coordinator request for Levanter cache processes, not by the model or GPU allocation.
- Commit Hash: live Marin bundle `49fabbb952043dbd41e4d2ab5cfe7c7df00f52bf`; MarinSkyRL launcher `086a4e248b4ec5bd9f09a28c37cb8d659034ad54`; cache sizing fix pending commit.
- Commands:
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-6acc0653/zephyr-levanter-cache-build-47468a9b-p0-a0`
  - `uv run infra/echo/cli.py search "Zephyr coordinator OOMKilled levanter cache build memory"`
  - `uv run pytest lib/levanter/tests/test_new_cache.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: pretraining used 4xGB200 for 16 steps; Levanter's cache workers request 32 GB, while their Zephyr coordinator inherited the 1 GB default. The fix explicitly requests 4 GB non-preemptible coordinator memory for every Levanter cache build, probe, and copy context.
- Result: pretraining finished at step 15 with loss `9.7033186`, 262,144 trained tokens, 117,831 tokens/s, and mean MFU `3.6576`; [W&B](https://wandb.ai/marin-community/marin-iceball-micro/runs/iceball-micro-pretrain) is terminal. The exact HF export is `s3://marin-us-east-02a/marin/checkpoints/iceball-micro-pretrain/2026.08.02/hf/step-15`. SFT then launched, but its one-shard chat cache coordinator was OOM-killed with exit 137 immediately after its worker registered. The Iris task requested 1 GB and made no optimizer progress. The complete Levanter cache suite passes 28 tests with the explicit coordinator size, and the changed-files lint/typecheck gate is green.
- Interpretation: the live timing and process boundary falsify a training-memory failure: the 4xGB200 SFT process remained alive while its CPU-only nested coordinator died during import and actor discovery. Prior Echo records independently report 1 GB coordinator OOMs and a healthy 4 GB allocation.
- Next action: deploy the tested cache sizing fix, resume the same artifact version, and verify the SFT cache and optimizer complete before entering RL.

### 2026-08-02 02:05 UTC - Cache recovery validated; native checkpoint handoff corrected

- Hypothesis: SFT can load the pretraining weights once the typed `LevanterCheckpoint` artifact resolves its checkpoint-series accessor instead of exposing its output root as an implicit filesystem layout.
- Commit Hash: live Marin bundle `b8f57a151d79697f1222f5f2f93cfd1a14b53339`; checkpoint-handoff fix pending commit.
- Commands:
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-2b9175fc/zephyr-levanter-cache-build-fe9de491-p0-a0`
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-2b9175fc/zephyr-levanter-cache-probe-5903f152-p0-a0`
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-2b9175fc/zephyr-levanter-cache-copy-1b2edb7b-p0-a0`
  - `uv run infra/echo/cli.py search "Levanter SFT initialize native checkpoint checkpoints subdirectory pretrain" --limit 20`
  - `uv run pytest tests/sft/test_hf_to_levanter.py tests/post_training/test_iceball_micro.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: all three Levanter cache coordinator phases request 4 GB non-preemptible memory. Native training artifacts write rolling checkpoints below their typed `checkpoint_dir` (`<artifact>/checkpoints`).
- Result: cache build, probe, and copy all succeeded with zero failures; the 9,500-row No Robots chat cache committed successfully. SFT then failed before step 0 because `LevanterCheckpointModel` passed `<pretrain artifact>` to `latest_checkpoint_path`, while the committed series is `<pretrain artifact>/checkpoints/step-15`. The model source now resolves an artifact handle through `LevanterCheckpoint.checkpoint_dir`; static checkpoint-series paths remain explicit. Nine focused SFT/workflow tests and the required lint/typecheck gate pass.
- Interpretation: stage chaining must pass typed artifact semantics, not assume an output root is itself a checkpoint. The existing `train_lm(init_from=...)` helper already follows this rule; the newer chat-SFT model source did not.
- Next action: deploy the tested handoff fix, verify SFT starts from the pretraining step-15 model with a fresh optimizer at step 0, then continue to packaged SkyRL.

### 2026-08-02 02:15 UTC - Native handoff validated; GPU SFT host memory sized

- Hypothesis: the exit-137 immediately after checkpoint discovery is host-memory exhaustion from the SFT launcher's default 4 GB request.
- Commit Hash: live Marin bundle `0f1f9306938f8077290e9823d0d6d94e0ac807bb`; GPU host-sizing fix pending commit.
- Commands:
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml job summary /power/iceball-micro-e2e-20260802/run_levanter_train_lm-80b38fea`
  - `uv run iris --cluster lib/iris/config/cw-us-east-08a.yaml query "SELECT task_id, attempt_id, state, exit_code, pod_name, node_name, terminal_reason, error FROM task_attempts ..." -f csv`
  - `uv run infra/echo/cli.py search "SFT resources_from_accelerator 4g memory OOM GB200 Levanter" --limit 20`
  - `gh issue view 7417 --repo marin-community/marin --json number,title,state,url,body,comments,labels`
  - `uv run pytest tests/sft/test_launcher.py tests/sft/test_hf_to_levanter.py tests/post_training/test_iceball_micro.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
- Config: the corrected handoff found `s3://marin-us-east-02a/marin/checkpoints/iceball-micro-pretrain/2026.08.02/checkpoints/step-15`; `resources_from_accelerator("4xGB200")` previously inherited 1 CPU, 4 GB RAM, and 16 GB disk. GPU SFT resources now scale by device: 8 CPU, 96 GB RAM, and 48 GB disk per GPU (32 CPU, 384 GB RAM, and 192 GB disk for iceball).
- Result: the SFT process loaded the committed cache, selected weights-only initialization with a fresh optimizer at step 0, and discovered the exact pretraining checkpoint. It was then SIGKILLed with exit 137 before emitting another log. This reproduces open [issue #7417](https://github.com/marin-community/marin/issues/7417), which documents the same default-4-GB GPU SFT failure and the same tested per-device resource schedule. Ten focused launcher/SFT/workflow tests and the required lint/typecheck gate pass.
- Interpretation: the timing, exit code, 595.8M-parameter checkpoint size, and exact default request match host OOM; this is not HBM pressure. Because resources are runtime arguments, correcting them does not fork artifact identity or invalidate successful upstream work.
- Next action: deploy the explicit resource schedule, verify native model load and SFT step 0, then continue to packaged SkyRL.

### 2026-08-02 03:40 UTC - SFT and packaged SkyRL training completed

- Hypothesis: explicit host resources and an exact launch/runtime contract let the same artifact graph cross the native-SFT to external-SkyRL boundary and recover from worker preemption without changing checkpoint identity.
- Commit Hash: Marin `3ed84be82`; MarinSkyRL `1b6c1873a27c6973a8647d90552b64dedc1c3fc9`.
- Config: SFT used 4xGB200 for eight optimizer positions (terminal step 7). RL used one 4xGB200 node, colocated policy and four single-GPU inference engines, batch size 16, four samples per prompt, eight terminal steps, and one shared checkpoint root across attempts.
- Result: SFT completed from the pretraining step-15 weights and exported `s3://marin-us-east-02a/marin/checkpoints/iceball-micro-sft/2026.08.02/hf/step-7`. Its terminal W&B summary reports loss `9.6932592`, 65,536 tokens, 73,786 tokens/s, and mean MFU `2.3483` at [run 2026.08.02](https://wandb.ai/marin-community/marin-iceball-micro/runs/2026.08.02). The packaged MarinSkyRL child resumed from durable step 6 after a planned preemption. A second attempt stopped making progress in policy mini-batch 14/16 with all policy GPUs busy; preempting only that task moved the same checkpoint to another node. The next attempt resumed step 6 and completed step 8, exporting `s3://marin-us-east-02a/marin/checkpoints/iceball-micro-rl/2026.08.02/exports/global_step_8/policy`. The terminal [RL W&B run](https://wandb.ai/marin-community/marin-iceball-micro/runs/92ver0qh) reports reward/pass@4 `0`, final loss `4.1423e-10`, entropy `11.4375`, KL `6.6277e-06`, 242.19 mean response tokens, and 14.15 seconds per step.
- Interpretation: the successful cross-node resume validates the terminal protocol, shared checkpoint root, attempt manifests, and exact policy export. The mini-batch stall was transient and node-specific rather than a deterministic package or checkpoint failure.
- Next action: serve the exact step-8 export once and drive both Evalchemy and Harbor from the same Marin evaluation artifact.

### 2026-08-02 04:55 UTC - Full Iceball workflow completed with durable evaluation records

- Hypothesis: Harbor's integration smoke can remain a real model evaluation while bounding a weak model that never emits EOS, so model quality produces a zero reward instead of an infrastructure failure.
- Commit Hash: Marin `1820053a8`; MarinSkyRL `1b6c1873a27c6973a8647d90552b64dedc1c3fc9`.
- Commands:
  - `uv run pytest tests/evaluation/test_harbor_trial_driver.py -m integration -q`
  - `uv run pytest tests/evaluation/test_model_config.py tests/evaluation/test_evaluation.py tests/evaluation/test_harbor_runner.py tests/rl/test_skyrl.py tests/post_training/test_iceball_micro.py -q`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run iris --cluster=cw-us-east-08a job wait /power/iceball-micro-e2e-20260802`
- Config: Evalchemy runs 16 GSM8K examples at zero shot with 256 generation tokens. The AIME integration smoke runs two Daytona trials through `SingleTurnAimeAgent`, which makes one 256-token OpenAI-compatible request, persists the raw response, converts the last AIME-sized integer (or `-1`) to `/app/answer.txt`, and allows Harbor's verifier to grade normally. The production `aime-harbor` policy remains on Terminus-2.
- Result: the parent [Iris workflow](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fpower%2Ficeball-micro-e2e-20260802) reached `succeeded`. Evalchemy wrote [its record](s3://marin-us-east-02a/marin/evals/iceball-micro/gsm8k-smoke,aime-smoke/2026.08.02/20260802-044938-iceball-micro-gsm8k-smoke-0358/record.json) with 16 samples and strict/flexible exact match `0.0`. Harbor wrote [its record](s3://marin-us-east-02a/marin/evals/iceball-micro/gsm8k-smoke,aime-smoke/2026.08.02/20260802-044938-iceball-micro-aime-smoke-2e1e/record.json) with two completed, zero-failure trials, accuracy/mean reward `0.0`, plus raw responses and per-trial verifier artifacts under its results prefix. Both model responses exhausted the bound by repeating `the` and contained no integer, so the zero scores are expected model behavior. Ten pinned-Harbor integration tests, 45 focused unit tests, Ruff, Black, Pyrefly, and repository structural checks pass.
- Interpretation: the completed run proves one Marin graph can transfer unchanged from programmatic artifact construction to an Iris CLI coordinator across random-init pretraining, SFT, external SkyRL, Evalchemy, and Harbor. The bounded agent belongs only to the smoke policy; full agentic evaluation retains upstream Harbor semantics.
- Next action: publish the operational incident record, run the branch-level review gate, update both PRs with terminal evidence, and seal the Weaver report.

### 2026-08-02 05:01 UTC - Recovery record published

- Result: the search-before-write audit found no prior SkyRL mini-batch-stall or Daytona-cleanup incident. [Echo wiki 69](https://echo.oa.dev/wiki/69) now records the package, resource, checkpoint, vLLM, Harbor, and live-retry failures, with the node-specific SkyRL stall explicitly left at unknown root cause.
- Next action: link the canonical record from the implementation PR and complete its final review and CI gates.

### 2026-08-03 19:24 UTC - Frozen standard-image SkyRL run completed

- Hypothesis: merged MarinSkyRL #284 plus the #296 frozen task bootstrap can run the same eight-step Iceball FSDP recipe without a SkyRL-specific image or private wheelhouse.
- Commit Hash: Marin `abfb223b3`; MarinSkyRL `aa88890c3de0ad455c7ca849fa6eb7eabc62c9d7`.
- Config: two standard `iris-task:f1b207d40` four-GB200 tasks; one node for the four-GPU policy/reference roles and one for four single-GPU vLLM rollout engines; PyTorch 2.11.0+cu129, stock vLLM 0.23.0, Ray 2.51.1; eight GRPO steps.
- Result: both tasks in [the clean Iris job](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fpower%2Fcheckpoints-iceball-micro-rl-2026.08.03-d6d59784d6e7) succeeded. Training completed eight steps in 131 seconds, wrote `checkpoints/global_step_8`, and exported `s3://marin-us-east-02a/marin/checkpoints/iceball-micro-rl/2026.08.03/exports/global_step_8/policy`. The terminal [W&B run](https://wandb.ai/marin-community/marin-iceball-micro/runs/o6up4alf) and typed `terminal.json` record the exact runtime, topology, inputs, job, checkpoint, and export.
- Interpretation: the common ARM64 FSDP/vLLM path is package-complete. The PyPI vLLM wheel's absent optional cuMem allocator blocks only colocated sleep/wake, so the supported disaggregated topology removes the last general custom-image dependency.
- Next action: run both evaluator mechanisms against the clean export and update PRs, issue #7920, and Weaver artifacts with terminal evidence.

### 2026-08-03 19:40 UTC - Harbor capability-proxy retry boundary corrected

- Hypothesis: the clean evaluation failure is a transient external capability-proxy response rather than a model, Daytona, or verifier failure.
- Result: Evalchemy completed its 16 GSM8K samples. Both concurrent Harbor sandboxes then received HTTP 503 from the Iris capability URL on their only model request; per-trial artifacts show the failure occurred before answer persistence or verification. Daytona's later `CancelledError` was teardown noise, not the trial root cause. Echo wiki 69 recorded a prior nonfatal form of that cleanup message but no request-retry fix.
- Change: `SingleTurnAimeAgent` now retries only timeouts, URL transport failures, and HTTP 408/425/429/5xx gateway statuses with bounded exponential delay. It still propagates non-retryable HTTP and response-schema errors immediately. A pinned-Harbor integration test observes a 503 followed by a successful graded response.
- Validation: the two agent integration tests pass, 32 focused evaluation/RL/workflow tests pass, and the required changed-files lint, formatting, and Pyrefly gate pass.
- Next action: let the coordinator's artifact retry finish or submit the fixed bundle, then record terminal Evalchemy and Harbor run IDs.

### 2026-08-03 20:00 UTC - Clean export evaluation completed after native-proxy recovery

- Hypothesis: the repeated HTTP 503 responses are caused by the public Iris capability proxy rather than Evalchemy, Harbor, Daytona, or the served model.
- Commit Hash: Marin `7782c53dd`; MarinSkyRL `aa88890c3de0ad455c7ca849fa6eb7eabc62c9d7`.
- Result: parent-controller logs identified the exact error: `Native proxy registry replacement failed; pausing native routing`, followed by `endpoint 26f54607-02e0-44e9-9440-fa6bb1aa6be2 address must be an absolute HTTP(S) URL`. The Python endpoint registry continued advancing while the Rust native proxy remained at generation 258518, so every capability route returned `endpoint registry unavailable`. The malformed leased endpoint expired without a controller restart. A later mapping update then installed a valid full snapshot and restored public routing. The workflow's next automatic attempt reached [terminal success](https://iris-cw-us-east-08a.oa.dev/#/job/%2Fpower%2Ficeball-micro-eval-fixed-20260803).
- Evaluation evidence: Evalchemy sent 16 successful model requests and wrote [run `20260803-195123-iceball-micro-gsm8k-smoke-6632`](https://evaldash.oa.dev/runs/20260803-195123-iceball-micro-gsm8k-smoke-6632) plus `s3://marin-us-east-02a/marin/evals/20260803-195123-iceball-micro-gsm8k-smoke-6632/record.json`. Harbor completed both AIME trials with zero infrastructure failures and wrote [run `20260803-195123-iceball-micro-aime-smoke-d2e4`](https://evaldash.oa.dev/runs/20260803-195123-iceball-micro-aime-smoke-d2e4) plus `s3://marin-us-east-02a/marin/evals/20260803-195123-iceball-micro-aime-smoke-d2e4/record.json`. Both mechanisms scored `0.0`, expected for the bounded random-init integration model.
- Interpretation: the Harbor retry is useful hardening for short transport failures, but it did not repair the persistent outage. The terminal workflow succeeded because the leased malformed endpoint expired, the native registry recovered, and the artifact-level retry launched a fresh shared inference/evaluation group. One invalid endpoint must not be able to pause all native proxy routing; that recovery failure belongs to Iris rather than this RL integration.
- Next action: publish the distinct Iris incident in Echo, link it from the final artifact, and keep the RL PR scoped to the completed package/workflow integration.
