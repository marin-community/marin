# Acceptance measurement: runbook

192 code-verified capabilities from the n=25 proposal run, one proposal each, authored and validated in three
matched arms. Wave 1 is the first 32 capabilities x 3 arms (96 allocations); wave 2 is the other 160 x 3 (480).

| Arm | Treatment (`arms/*.json`) |
| --- | --- |
| A | the stack runtime (#10301), every key set explicitly; unchanged-resubmit refusal and the identical-error detector are on in every arm |
| B | A + strictness guidance and reference kit, `embed_file` and `run_probe_battery` tools, battery enforced at submit, witness replay, defect-class feedback |
| C | B + saturated accept: a band-only TOO_EASY round with 8/8 solved and a passing FA search is accepted (`band_label: saturated` in `acceptance-review.json`) |

`treatment.py` lists every key with the surfaces it may change. `dry-run` renders all three arms with no model
and fails if a pair differs anywhere it should not, or fails to differ where it must.

## Before launch

1. **The n=25 run has finished**, and every capability dir under its `run/` root has `result.json`. A sample
   drawn earlier is incomplete; the driver refuses it unless passed `--allow-incomplete-population`.
2. **The relays route both models.** Ask the `glm_infer` session whether relays g and f route
   `deepseek-v4.1-flash` as well as `glm-5.3`, and whether DeepSeek accepts the `reasoning_effort` template kwarg.
   Each job's preflight resolves the relay for every model it uses and exits before any allocation starts.
3. **`lib/iris` is less than 14 days old** (the hub rejects an older `job run` client) and HEAD is committed:
   the manifest records `git rev-parse HEAD`, and `launch.sh` refuses a dirty tree.

## Launch

```bash
A=experiments/post_training/capability_driven_envs/acceptance_measure/launch.sh
$A sample     # draw sample.json, freeze it at $ROOT/sample.json (refuses to overwrite)
$A dry-run    # three arms on the first 3 sampled proposals; writes diff.json; no model
$A wave1      # 96 jobs: am-w1-<arm>-r<rank>-g1, one allocation each
```

Each job is one allocation on `cw-rno2a`: 32 CPU, 128 GB, 300 GB disk, non-preemptible, `--max-retries 2`,
3-day timeout. That is the sizing the factorial author parents ran at: one PC parent OOM'd at 128 GiB, and
parents grew to 40 GiB. Even ranks use relay g, odd ranks relay f. `$ROOT` defaults to
`s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261102`; override it with `ROOT=`.

**Resume and relaunch.** Each allocation keeps its tree, records and attempts under
`$ROOT/allocations/<arm>/<capability>/`. `SEALED.json` is written last, and only for accepted, rejected or
abandoned. A sealed allocation never reruns. An unsealed one restores its uploaded tree and continues,
for up to 3 attempts. To relaunch failed jobs, run with a new generation (`GEN=g2 $A wave1`): sealed
allocations exit at once, and the old jobs keep their post-mortems. A held allocation (the coordinator holds a
workpiece on infrastructure failure, or on an Accept outside IN_BAND) stays unsealed.

## Monitor and read

```bash
uv run --frozen iris --cluster=marin job list --prefix /muchanem/am-w1 --limit 500
uv run --frozen iris --cluster=marin job logs /muchanem/am-w1-b-r007-g1 --max-lines 400 --since-seconds 900
$A reduce 1   # funnel, acceptance per arm, tokens by stage, attribution, McNemar pairs; from S3 records only
```

## Stop rule and gates

- **No early stopping on outcomes.** Read the reducer descriptively at 48 and at 96 terminal pairs only.
- **Gate before wave 2.** All five must hold:
  - the reducer reproduces the funnel from the records;
  - a seal survives a deliberate restart (cancel one job, then relaunch it with `GEN=g2`);
  - `acceptance-shadow.json` exists for every validated round;
  - completion (sealed / allocated) is at least 85%;
  - charged components show 0 typed infrastructure failures.
- **Wave 2** goes in slices (`$A wave2 32 96`, then `$A wave2 96 192`) sized to what the fleet takes.
- **Fleet:** the shed rule in `CLAUDE.md` section 11 applies. `glm_infer` owns capacity; message it before each
  wave. Token alarms report and never gate.

## Cost

An allocation costs 15-25M tokens (uncached input plus output, author included). Wave 1 is about 1.4-2.4B tokens;
the full run (576 allocations) is about 9-14B.

## Not built

- **Arm A is the stack runtime**, not the PC fork.
- **The kit reaches the author through files and tools**, because the stack author has no shell.
- **Finders get the contract only through the instruction.** An explicit contract block would need PR7's brief.
- **The defect class comes from a keyword heuristic.** The shadow judge's `defect_class` is the validated enum.
- **Missing pieces:** constrained decoding, the phrase-loop detector, placeholder-evidence rejection, heldout
  rows and the PC-record reducer check.
- **Progress is uploaded only when a child ends**, so a preemption loses the in-flight attempt.
- **The identical-error hold is a transient infrastructure failure**, so the coordinator retries it up to 4 times.
