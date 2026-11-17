# Forge tasks

A public Marina applet for the forge main-split release: every selected task with its prompt, grader, files,
controls, validation history and the eight GLM-5.3 confirmation trials.

## Data layout

The applet is fast because nothing on the first paint touches S3, and nothing later parses a forge event log.

| What | Where | Size | Loaded |
|---|---|---|---|
| Index: one columnar row per task, catalog names, pool counts | `applet/data/{meta,tasks}.json`, shipped in the package | ~1 MB | at start |
| Catalog detail per subject (outcome, includes, sample tasks) | `applet/data/subjects/<D..>.json` | small | on a subject or skill page |
| Task record: prompt, grader, files, controls, rounds, decisions | `<export>/app/<snapshot>/detail/<SUBJECT>.json.gz` | one per subject | on hover or open of a task |
| Trimmed transcripts for all tasks of a skill | `<export>/app/<snapshot>/trials/<capability>.json.gz` | one per skill | with the task record |
| One full transcript | `<export>/records/<id>/trials.json.gz` | one per task | when a trial is opened |

The backend (`applet/server/app.py`) serves those S3 objects pre-gzipped from in-process LRU caches with
`immutable` cache headers keyed by the snapshot, so a browser fetches each one once. The client memoizes every
fetch and prefetches a task's record when the pointer rests on it.

## Rebuilding

1. Export every accepted confirm workpiece to per-task records (`experiments/forge/export_tasks.py` on the forge
   branch; one Iris job per shard, already-exported tasks are skipped).
2. Select the release: `uv run experiments/post_training/task_curation/forge_release.py --records <export>/records
   --target 4000 --output <dir>`.
3. Build the applet data: `uv run infra/marina/apps/forge_tasks/build_data.py --export <export> --release <dir>
   --snapshot <YYYYMMDDTHHMMZ>`. This writes `applet/data/` and uploads the detail and trial shards.
4. Preview with `uv run marina release infra/marina/apps/forge_tasks/applet --local --json`, then release with
   `--mode public`.

`applet/data/` and `applet/dist/` are build outputs and are not committed.
