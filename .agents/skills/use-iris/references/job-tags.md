# Iris job tags

Tag every root Iris job with `--project`, `--category` and `--description`
(SDK: `IrisClient.submit(tags=job_pb2.JobTags(...))`). The tags go to Finelog
`iris.controller_db__jobs.tags` and are kept permanently: never put credentials
or sensitive content in them.

## Category

The kind of work, from a fixed list; `iris job run` and the controller refuse
any other value. On a root job it is what the whole job tree is for. A child job
that sets no category takes its parent's.

- `pre-training`: base or MoE language-model training.
- `sft`: supervised and preference fine-tuning (SFT, DPO).
- `rl`: reinforcement learning runs.
- `evaluation`: benchmark and eval runs, and the harness or scoring job itself.
- `data-processing`: download, normalization, dedup, decontamination,
  tokenization and synthetic data pipelines.
- `inference`: inference servers and their workers; on a root job, a standing
  service.
- `training`: a training job under a pre-training or SFT root.
- `testing`: canaries, ferries, CI and smoke runs.
- `development`: dev sessions, interactive and one-off engineering runs, ad hoc
  benchmarks.
- `infrastructure`: platform services, dashboards and maintenance jobs.

A canary or ferry that runs an evaluation is `testing` at the root; the
evaluation job under it is `evaluation`.

## Project

The effort, as free text of up to 100 characters; usually the `experiments/`
area or library. Reuse a value below when one fits, so compute reports group
cleanly, and add a new one only for a new effort.

- `grug`: Grug/Grugformer model variants (`experiments/grug`).
- `moe-hero`: large MoE hero runs (`experiments/grug/moe_hero_*`, `experiments/june_tpu_67b_a2b`).
- `rl`: SkyRL and Snowball RL (`experiments/post_training`).
- `sft`: supervised fine-tuning (`experiments/sft`, `experiments/post_training/*sft*`).
- `swe-agentic`: SWE and agentic task curation and rollouts (`experiments/post_training/task_curation`, `experiments/rollout_data`).
- `evaluation`: the eval catalog and campaigns (`experiments/evaluation`).
- `datakit`: the data pipeline (`experiments/datakit`).
- `data-belay`: dataset-value tests (`experiments/data_belay`).
- `datasets`: dataset registration and tokenization (`experiments/datasets`).
- `taskforge`: task generation (`lib/taskforge`).
- `inference`: vLLM serving and parity (`experiments/evaluation/serve`).
- `ferries`: canary and daily ferries (`experiments/ferries`).
- `scaling-laws`: reference scaling suites (`experiments/references`).
- `weight-merging`: weight merging (`experiments/weight_merging`).
- `mlx-export`: MLX export and parity (`experiments/mlx_export`).
- `kernels`: Pallas kernels and benchmarks (`experiments/benchmarks`).
- `iris`, `zephyr`, `levanter`, `fray`: library development and smoke runs.
- `tutorials`: onboarding and tutorial runs (`experiments/tutorials`).
- `amd-hpcfund`: the AMD HPC Fund collaboration (`experiments/amd/hpcfund`).

## Description

Free text of up to 1000 characters on what the job does.
