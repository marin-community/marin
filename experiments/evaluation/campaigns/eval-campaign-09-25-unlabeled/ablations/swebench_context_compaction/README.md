# SWE-bench context-compaction ablation

This 4 × 4 grid evaluates SWE-bench Verified with mini-swe-agent 2.1.0. It varies total context,
per-response output capacity, and agent-history summarization while holding every other model and
benchmark setting fixed.

| Arm | Total context | Output cap | Summarization |
|---|---:|---:|---|
| `65k-16k-no-summary` | 65,536 | 16,384 | off |
| `131k-16k-no-summary` | 131,072 | 16,384 | off |
| `32k-8k-summary` | 32,768 | 8,192 | 8,192 accumulated history characters |
| `65k-16k-summary` | 65,536 | 16,384 | 8,192 accumulated history characters |

The summary threshold counts the provider-facing serialized history after the original system and
task messages. Compaction preserves the newest complete turns that fit half of the threshold and
uses the other half for a rolling summary. The saved trajectory retains the full unabridged message
history and records compaction telemetry under `info.context_compaction`.

`generate_configs.py` deterministically materializes all exact model-serving and Harbor configs.
The generated files are checked in and are the launch inputs; `launch.py` does not alter serving or
benchmark semantics.

## Fixed-runtime recovery

The original no-summary runs used Harbor `39267af2`, whose Daytona cancellation
cleanup could leave mini-swe-agent running after its deadline. Recovery uses
Harbor `2993c283`. `configs/recovery/<model>/<arm>/manifest.json` records the
original result URIs and SHA-256 checksums, retained tasks, and recovery tasks.
Its sibling `harbor.yaml` changes only task selection from the full-arm config.
The runtime revision is pinned in `config/external/harbor/pyproject.toml` and
its lockfile. Each run records the resolved revision in
`harbor_resume_identity.json`.

Retention follows Harbor's native scoring classification, excluding attempts
that reached the old agent deadline and nonzero-exit failures before agent
execution. The generator checks identical 500-task dataset descriptors and
rejects duplicate source task records.

`prepare_recovery.py` generates these files but does not stop or submit jobs.
Generate the final manifest only after the source root and its inference child
have stopped. For example, from the repository root:

```bash
ablation=experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/swebench_context_compaction
uv run python "$ablation/prepare_recovery.py" \
  --source-results "$source_results" \
  --base-config "$ablation/configs/harbor/65k-16k-no-summary.yaml" \
  --destination "$ablation/configs/recovery/qwen36/65k-16k-no-summary"
uv run python -m experiments.evaluation.cli launch \
  --model-config "$ablation/configs/models/qwen36/65k-16k-no-summary.yaml" \
  --harbor-config "$ablation/configs/recovery/qwen36/65k-16k-no-summary/harbor.yaml" \
  --federated_cluster cw-rno2a --priority interactive --no-wait
```

Set `source_results` to the original run's results URI. Each checked-in manifest
contains this URI for its model and arm. For another arm, use its corresponding
model config and recovery directory. Validate the launch command with
`--dry-run` instead of `--no-wait`, then commit the generated files before
launching. Do not pass a task limit or resume into the
old result root. For the full-arm score, combine the retained results with
recovery results by task ID exactly once over the original 500-task cohort.
The recovery subset's aggregate alone is not a full-arm score.
Audit a recovery root with:

```bash
uv run python "$ablation/combine_recovery.py" \
  --manifest "$ablation/configs/recovery/qwen36/65k-16k-no-summary/manifest.json" \
  --recovery-results "$recovery_results" --output combined.json
```

`recovery_results` is the replacement run's `s3://.../results` URI, containing
`harbor_jobs/` and `harbor_resume_identity.json`. The audit reads run artifacts
and writes a local report. It checks the manifest's Harbor revision, launched
task selection, retained `result.json` checksums, and task uniqueness across
both inputs. Each task has one trial. The report includes their combined mean,
missing and unscored counts, and each trial's URI, checksum, and score.
`all_tasks_scored` becomes true only when all 500 manifest tasks have a score.
The command does not publish to EvalDash. Verify Iris job completion separately.

The audit also reads `service_outages.json`. It excludes any completed agent
execution that overlaps a recorded outage whose `results_root` contains the
trial's source URI, regardless of
reward. These exclusions are analysis metadata; raw rewards and native error
types remain unchanged. The report records the outage file's checksum and
counts excluded tasks as unscored. An outage with `finished_at: null` remains
open. Set its end only after inference requests have been verified to succeed
again, then rerun the audit.
Use `--service-outages PATH` to audit another recorded outage set.
