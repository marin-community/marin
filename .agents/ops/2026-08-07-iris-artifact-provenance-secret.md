---
date: 2026-08-07
system: iris
severity: near-miss
resolution: mitigated
pr: none
issue: none
---

# TL;DR

- Three Grug recovery controllers received `WANDB_API_KEY` through an explicit `-e` argument.
- Seven completed `.artifact.json` records copied that unredacted argument into `provenance.command_line`; no credential value is reproduced here.
- Iris API responses redacted both job environment values and `submit_argv`, and all 18 executor metadata/status sidecars were clear.
- Further launches with explicit secret arguments were stopped. Iris already auto-injects `WANDB_API_KEY` from the submitter environment, so the safe immediate path is to omit the `-e` pair.
- Artifact provenance still needs a code fix: `MARIN_PROVENANCE` captures raw `sys.argv` independently of Iris's redacted `submit_argv`.

# Original problem report

Terminal checkpoint verification opened a completed `.artifact.json` record and revealed that `provenance.command_line` contained the unredacted value following `-e WANDB_API_KEY`. The three affected controller IDs were `/dlwh/grug-xem-merge-identity-20260807`, `/dlwh/grug-xem-merge-native-20260807`, and `/dlwh/grug-xem-merge-spectral-20260807`.

# Investigation path

1. The completed output prefixes were enumerated without reading checkpoint tensors. Seven `.artifact.json` records belonged to work produced by the three controllers: three conversion records, three Stage-A records, and the selected native Stage-B record.

2. Each record was classified with a bounded `jq` predicate that reported only absent, redacted, or unredacted. All seven contained an unredacted value after the sensitive variable name. No value was printed or copied into the repository.

3. The three controller `GetJobStatus` responses were classified the same way. Both `request.environment.env_vars` and `request.submit_argv` returned `[REDACTED]`, confirming that the controller's public read path was not the leak.

4. Nine `.executor_info` and nine `.executor_status` sidecars under the three branch prefixes were checked for the sensitive variable name. None mentioned it.

5. `lib/iris/src/iris/cluster/redaction.py:40-93` showed that Iris separately redacts environment maps returned by the controller and explicit `-e` values captured in `submit_argv`. `lib/iris/src/iris/cluster/controller/service.py:1939-1945` applies the environment redactor before returning job status.

6. `lib/iris/src/iris/cluster/types.py:752-763` showed that task environments also receive `MARIN_PROVENANCE` from `launch_provenance().to_json()`. `lib/rigging/src/rigging/provenance.py:118-127` builds that record from raw `sys.argv`, so it bypasses `redact_submit_argv()` in `lib/iris/src/iris/cli/job.py:1053`.

7. `lib/iris/src/iris/cli/job.py:185-187` and `lib/iris/src/iris/cluster/types.py:752-760` confirmed the safe immediate submission path: `WANDB_API_KEY` is automatically copied from the submitter environment when no explicit `-e` argument is present.

# User course corrections

- The operator stopped further launches with direct secret expansion and requested a read-only scope audit. This prevented additional provenance records from being written before the capture path was understood.
- The operator prohibited deleting or rewriting GCS artifacts and prohibited a public issue. The audit therefore classified records in place and kept the incident record inside the repository.

# Root cause

Iris has two independent launch-history paths. `redact_submit_argv()` sanitizes the request stored by the controller, but `EnvironmentSpec.to_proto()` publishes `launch_provenance()` through `MARIN_PROVENANCE`. `Provenance.capture()` populates `command_line` from raw `sys.argv`. An explicit `-e WANDB_API_KEY <value>` therefore remained in the provenance inherited by Marin workers and was persisted into every completed artifact record.

The existing redaction tests cover controller request shapes, not the `MARIN_PROVENANCE` record consumed by `marin.execution`. This allowed controller status to look safe while completed artifact metadata still retained the argument.

# Fix

The operational mitigation is to leave `WANDB_API_KEY` in the submitter environment and omit the explicit `-e WANDB_API_KEY ...` pair. Iris auto-injects the variable into the job environment without placing its value in `sys.argv`. No affected GCS record was modified or deleted.

A durable code fix should construct `MARIN_PROVENANCE.command_line` from the already-redacted `submit_argv`, or apply an equivalent redactor before serializing provenance. Add an end-to-end regression that submits `-e WANDB_API_KEY <sentinel>` and asserts that neither the returned request nor inherited `MARIN_PROVENANCE` contains the sentinel.

# How OPS.md could have shortened this

- In `lib/iris/OPS.md`, extend the task-launch guidance with: "Standard credentials such as `WANDB_API_KEY` and `HF_TOKEN` are auto-injected from the submitter environment. Do not place secret values in explicit CLI arguments; inspect `GetJobStatus` and downstream artifact provenance separately when auditing exposure."
- Add a read-only audit example that classifies a sensitive field as absent, redacted, or unredacted without printing the value. This general pattern applies to any controller or artifact metadata incident.

# Artifacts

- `gs://marin-us-central1/grug/expert_merge/d512/identity/converted/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/identity/stage-a/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/native/converted/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/native/stage-a/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/native/stage-b/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/spectral/converted/2026.08.06/.artifact.json`
- `gs://marin-us-central1/grug/expert_merge/d512/spectral/stage-a/2026.08.06/.artifact.json`
