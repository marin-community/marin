---
date: 2026-05-02
system: iris
severity: degraded
resolution: investigating
pr: none
issue: "https://github.com/marin-community/marin/issues/5216, https://github.com/marin-community/marin/issues/5374"
---

### TL;DR

- Two `1e21 p67m33` recovery launches did not resume the failed W&B/checkpoint namespaces. They created new namespaces instead: `lr0.5` moved from `114e49` to `fdc4ebf1`; `lr0.67` moved from `ecbd27` to `99752407`.
- This was not only a W&B display issue. The Levanter checkpoint base paths in W&B config point at different GCS trees, so training repeated already-completed steps.
- The old `lr0.67` run had reached step `2651` / progress `0.601`, not the `5500-6500` range claimed in one later note. The old `lr0.5` run had reached step `3541` / progress `0.803`.
- The immediate operator error was relaunching without forcing the original output path and original W&B id, then not checking startup logs for the same run id plus `Resuming training from step ...`.
- The underlying footgun is Marin executor identity: the human-readable step name is not the run id. The output hash suffix is part of the checkpoint/W&B identity.

### Original problem report

The user saw repeated W&B rows for `1e21` runs and noticed that the newer
`delphi-1e21-p67m33-9p25b-lr0.67-99752407` curve was behind the crashed
`delphi-1e21-p67m33-9p25b-lr0.67-ecbd27` curve at the same wall-clock point.
The exact symptom was:

```text
for example this run thats 997 whatever is completely re-doing work we did already
```

### Investigation path

1. The first check compared local W&B configs under `midtrain_wandb_data/runs`.
   The configs proved that `ecbd27` and `99752407` had different
   `trainer.checkpointer.base_path` values:
   `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/checkpoints`
   versus
   `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-99752407/checkpoints`.
   The same split existed for `lr0.5`: `114e49` versus `fdc4ebf1`.

2. The W&B histories showed that the new runs started from `global_step=0`,
   `run_progress=0`, and `eval/loss=2.044572114944458`, the same initial eval
   value as the crashed original runs. That ruled out a pure W&B-row problem.

3. GCS checkpoint listings showed the amount of work already available in each
   namespace at investigation time:

   ```text
   ecbd27 permanent checkpoints: 441, 882, 1323, 1764, 2205, 2646
   99752407 permanent checkpoints: 441, 882, 1323, 1764, 2205, 2646, 3087
   99752407 temp checkpoints: 643, 3149, 3217

   114e49 permanent checkpoints: 441, 882, 1323, 1764, 2205, 2646, 3087, 3528
   fdc4ebf1 permanent checkpoints: 441, 882, 1323, 1764, 2205, 2646, 3087
   fdc4ebf1 temp checkpoints: 3382
   ```

4. Local W&B summaries quantified the wasted prefix at the time of the data
   dump:

   ```text
   ecbd27:  global_step=2651, run_progress=0.600997506234414, state=crashed
   99752407: global_step=2199, run_progress=0.4985264112446157, state=running

   114e49:  global_step=3541, run_progress=0.8027658127408751, state=crashed
   fdc4ebf1: global_step=2457, run_progress=0.5570165495352528, state=running
   ```

5. The logbook showed the correct pattern had been understood for `1e21
   p67m33/lr0.33`: only that job was relaunched with
   `MIDTRAIN_OUTPUT_PATH_OVERRIDE=gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.33-ab4e64`.
   The `lr0.5` and `lr0.67` recovery launches later did not preserve their
   original failed namespaces.

6. A later note claimed the old `ecbd27` run probably reached step `5500-6500`.
   That was wrong for this run. W&B and GCS both put the old `ecbd27` progress at
   about step `2650`. The root cause analysis still held; the waste estimate did
   not.

### User course corrections

- The user rejected treating duplicate W&B rows as harmless visualization noise.
  That was correct because local W&B config and GCS checkpoint trees showed
  different Levanter checkpoint namespaces.
- The user insisted that future recovery must always resume the old run. That
  was the right invariant: for these Delphi midtraining runs, a recovery is not
  successful until startup logs show the old output path, old run id, and
  `Resuming training from step ...`.
- The user pushed back on broad architectural fixes while asking for immediate
  postmortem. That kept this investigation focused on the operational failure:
  a bad recovery command and a missing verification step.

### Root cause

The immediate root cause was a bad recovery contract. The recovery launch used a
new Marin-derived output hash instead of the original failed run's exact
checkpoint namespace. Since Levanter uses `trainer.checkpointer.base_path` as
the source of checkpoint discovery, the new job only saw checkpoints under its
new tree. It did not see `ecbd27` checkpoints when running as `99752407`, or
`114e49` checkpoints when running as `fdc4ebf1`.

The deeper cause is that Marin's human-readable `ExecutorStep` name is not the
checkpoint identity. The suffix derived from executor/StepSpec hash is part of
the run id and GCS path. Setting `WANDB_RESUME=allow` is insufficient unless
`WANDB_RUN_ID` is the original W&B id and `MIDTRAIN_OUTPUT_PATH_OVERRIDE` is the
original output path. If either value points at the newly derived hash, W&B and
Levanter both treat the recovery as a different run.

### Fix

A code-level guard was added after this postmortem. For
`experiments/exp_delphi_math_10b_midtrain.py`, recovery now uses one source of
truth:

```bash
# Before recovery, find the original failed namespace.
OLD_RUN_ID=delphi-1e21-p67m33-9p25b-lr0.67-ecbd27
OLD_OUTPUT_PATH=gs://marin-us-east5/checkpoints/${OLD_RUN_ID}

# Recovery launch must force only the old output path and an expected checkpoint floor.
-e MIDTRAIN_RESUME_OUTPUT_PATH "${OLD_OUTPUT_PATH}" \
-e MIDTRAIN_EXPECT_RESUME_MIN_STEP 2600
```

The script derives `RUN_ID`, `WANDB_RUN_ID`, and the executor output override
from `MIDTRAIN_RESUME_OUTPUT_PATH`, refuses legacy
`MIDTRAIN_OUTPUT_PATH_OVERRIDE`, and checks permanent/temp checkpoint discovery
before `executor_main` runs. `MIDTRAIN_ALLOW_EMPTY_RESUME=1` is the only escape
hatch, and is intended only for namespace-preserving restarts before any
checkpoint exists.

Generic Marin training also now rejects executor-backed training when the
resolved run id does not match the output path basename. The startup
verification must still include all three runtime signals:

```text
Using output path: gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27
Using run ID: delphi-1e21-p67m33-9p25b-lr0.67-ecbd27
Resuming training from step ...
```

If `Resuming training from step ...` is absent, the job should be treated as a
failed recovery even if it is actively training.

### How OPS.md could have shortened this

- Add an Iris OPS.md subsection for training recovery verification. Suggested
  text: "For Marin training jobs, do not judge recovery by Iris job name. Inspect
  training startup logs for the concrete output path, W&B run id, and checkpoint
  resume step. If the output hash changed, the job is a new training run."
- Add a command pattern for metadata-only checkpoint namespace comparison:

  ```bash
  gcloud storage ls gs://marin-<region>/checkpoints/<run-id>/checkpoints/
  gcloud storage ls gs://marin-<region>/tmp/ttl=14d/checkpoints-temp/*<run-id>*/
  ```

  The signal it unblocks is "is this recovery seeing the same checkpoint tree?"
- Add a W&B config check for Levanter runs:

  ```bash
  uv run python - <<'PY'
  import json, pathlib
  cfg = json.loads(pathlib.Path("midtrain_wandb_data/runs/<run-id>/config.json").read_text())
  print(cfg["trainer"]["id"])
  print(cfg["trainer"]["checkpointer"]["base_path"])
  PY
  ```

  This is a recurring diagnostic for any W&B duplicate-run incident.

### Artifacts

- Logbook: `.agents/logbooks/midtraining_delphi.md`
- W&B local dump:
  - `midtrain_wandb_data/runs/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/config.json`
  - `midtrain_wandb_data/runs/delphi-1e21-p67m33-9p25b-lr0.67-99752407/config.json`
  - `midtrain_wandb_data/runs/delphi-1e21-p67m33-9p25b-lr0.5-114e49/config.json`
  - `midtrain_wandb_data/runs/delphi-1e21-p67m33-9p25b-lr0.5-fdc4ebf1/config.json`
- Checkpoint namespaces:
  - `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/checkpoints/`
  - `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-99752407/checkpoints/`
  - `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-114e49/checkpoints/`
  - `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.5-fdc4ebf1/checkpoints/`
