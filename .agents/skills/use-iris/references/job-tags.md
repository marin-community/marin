# Iris job tags

Tag every root Iris job with `--category` and `--description`
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

## Description

Free text of up to 1000 characters on what the job does.
