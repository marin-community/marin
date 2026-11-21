# Capability-driven environments

This directory runs Taskforge over capability-driven task proposals: each proposal becomes a verified RL task
(accepted, abandoned or held) under the library defaults and a policy overlay. `launch.py` submits one Iris job per
range of proposal-key buckets, each running `taskforge.driver run`; everything else (authoring, validation, review,
confirmation, checkpoints, resume) is the library's. How the pipeline behaves, its policy keys, outputs and
`report` are in the [Taskforge README](../../../lib/taskforge/README.md).

## Launch

From the repository root, with the policy overlay already on S3:

```bash
uv run --frozen experiments/post_training/capability_driven_envs/launch.py \
  --proposals s3://<bucket>/<proposals>/sample.json --ranks "$(seq -s, 0 31 | sed 's/,$//')" \
  --output s3://<bucket>/<root>/out --policy s3://<bucket>/<root>/policy.json --job-prefix <prefix> \
  --drivers 7 --concurrency 4 --cpu 8 --memory 32GB --disk 100GB
```

`--drivers N` splits the 4096 buckets evenly; `--driver-range a:b` launches only those driver indices. To replace
drivers 24-31 of a 32-driver fleet with 16 drivers, cancel them and launch `--drivers 64 --driver-range 48:64` under
a new job prefix: the same 3072:4096 span, with no other driver, output path or resume behaviour changed. Each job
is `/<user>/<prefix>-<driver-index>` on `--target-cluster` (default `cw-rno2a`), with `--max-retries 3`,
`--sync-package marin-taskforge`, and `OMP_NUM_THREADS`, `RAYON_NUM_THREADS`, `CARGO_BUILD_JOBS` and
`CMAKE_BUILD_PARALLEL_LEVEL` pinned to `--cpu`. `--resume-held` is forwarded to the drivers only when given; use it
under a new job prefix after an infrastructure outage.

## Environment

Export in the launching shell: `CW_KEY_ID`, `CW_KEY_SECRET` (your own CoreWeave storage key), `GLM_API_KEY`, and
`PARALLEL_API_KEY` (optional, the author's web tools), plus exactly one of `GLM_RELAY_JOB` (the Iris relay job) or
`GLM_BASE_URL` (a direct endpoint). The keys go to Iris as `--secret NAME env:NAME`, never as values on the command
line; only the endpoint variable is passed by value.

## Exit codes

Each driver runs every proposal in a `one` child. It prints `OUTCOME <label> exit=<code> proposal=<path>` for every
nonzero exit except a probe's 10, which it consumes silently. `uv run --frozen python -m taskforge.driver --help`
prints the same list.

| code | meaning | what to do |
|---:|---|---|
| 3 | another writer owns the root | nothing |
| 4 | the root has another journal format | rerun at a fresh output prefix (no migration ships in this repository) |
| 5 | the root's journal is corrupt | move the root aside, then rerun |
| 75 | held until the decision model answers again | nothing: every later launch enters it, and the driver parks it until the model's health route reports healthy |
| 10 | a `--probe` child's own decision-model reader read a valid answer (gateways without the health route); no OUTCOME line | nothing: the driver releases the parked proposals |

The driver job fails when any child exits with a code other than 0, 3 or 75, apart from the probe's 10; a 10 from
any other child is a failure.

Output roots follow the format fence and the rollout rule in the
[Taskforge README's Outputs section](../../../lib/taskforge/README.md#outputs): a root in another journal format
exits 4 untouched, and a root written by this driver is never launched by an earlier one. No migration for Taskforge
journals ships in this repository, so an exit-4 root, like a rollback of the driver, takes a fresh `--output` prefix.
