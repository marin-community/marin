# Delphi Baseline Mixtures for Issue 6607

## Scope

Train objective-agnostic proportional and UniMax-8 baselines over the 39 Dolma3/Dolmino top-level buckets on a small Delphi scaling ladder, then evaluate the resulting checkpoints with the full downstream stack including DCLM Core v2.

- GitHub issue: https://github.com/marin-community/marin/issues/6607
- Fieldbook experiment: `exp_01kvvvv6zxrf0j7tkp4f7k6y66`
- Launcher: `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py`

## 2026-06-23 Launcher Draft

Decisions:

- Use the Dolma3/Dolmino top-level bucket family: 39 buckets, not the 167-partition production-swarm split.
- Use two phases with identical weights in both phases for both baselines.
- Use proportional weights and UniMax with a global 8-epoch cap.
- Compute UniMax-8 weights against the fixed historical top-level target budget, not the realized train-token budget, so the materialized epoch cap is invariant across scaling rungs and compatible with the existing 39-bucket swarm convention.
- Keep the `exp1337_delphi_suite.py` CompletedAdamH/Delphi model, optimizer, and mesh logic; only replace the data mixture.
- Pin live training to `us-east5/us-east5-a` and `gs://marin-us-east5`.
- Skip lm-eval harness during training; downstream evals should run after final checkpoint/HF export.

Planned training matrix:

| Target FLOPs | TPU | Batch size | Mixtures |
|---:|---|---:|---|
| `3e18` | `v5p-8` | 128 | proportional, UniMax-8 |
| `2e19` | `v5p-16` | 128 | proportional, UniMax-8 |
| `3e20` | `v5p-32` | 256 | proportional, UniMax-8 |
| `1e21` | `v5p-64` | 512 | proportional, UniMax-8 |

Validation:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes.
- Fieldbook has 8 active datapoint runs.

Simulated epoching convention:

- Set `target_budget=TARGET_BUDGET_DOLMA3_COMMON_CRAWL` (`6.325T`), matching the existing 39-bucket Dolma3/Dolmino swarm convention.
- Set `experiment_budget=realized_train_tokens` per scaling rung.
- This slices every bucket by the same ratio `realized_train_tokens / TARGET_BUDGET_DOLMA3_COMMON_CRAWL`; for fixed weights, each bucket's materialized epoch count is `w_i * TARGET_BUDGET_DOLMA3_COMMON_CRAWL / bucket_size_i`, independent of model scale.
- For proportional this is about `0.905` materialized epochs for every bucket because the historical target is the Common Crawl subtotal rather than the full `6.986T` top-level pool. For UniMax-8, weights are computed at the same fixed target budget, so small buckets are capped at 8 total materialized epochs at every scaling rung.

Current blocker:

- Local submission/dry-run through `uv run` is blocked by the repository dependency-resolution conflict between `tpu-inference==0.22.1` requiring `numba==0.62.1` and the workspace `vllm` extra requiring `numba==0.65.0`. There is no standalone `iris` binary on this machine. Submit from a known-good Marin environment or after resolving the local `uv` environment.

## 2026-06-24 CC Launcher Review

Claude Code was invoked via `claude -p` with `env -u ANTHROPIC_API_KEY`, Opus 4.8, and read-only repo access. The long tool-based review did not return a clean final answer before stalling, but the transcript repeatedly identified a concrete launcher blocker: `IsoFlopAnalysisConfig.training_runs=[run.as_input_name() for run in adamh_training]` made the historical `nemotron-completed-adamh` sweep executable dependencies of this east5 launcher. Since those source runs were historical v4/default-prefix runs, a live east5 submission could attempt to schedule or resolve those upstream sweep steps under `gs://marin-us-east5/checkpoints/isoflop/...`.

Patch:

- Added `_completed_adamh_metric_sources()` so the isoflop analysis uses `run.as_input_name().nonblocking()` for all historical CompletedAdamH metric sources.
- This keeps the analysis step as a normal blocking dependency for the new Delphi baseline trainings, but prevents old isoflop training steps from being scheduled by this launcher.
- The analysis step still writes its own output under the active east5 `MARIN_PREFIX`; missing small `tracker_metrics.jsonl` files can be backfilled from W&B by run name rather than by cross-region GCS reads.

Validation:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes after the nonblocking-dependency patch.

## 2026-06-24 CC Retry Review After Import Fix

Claude Code was invoked via `claude -p` / resumed Opus 4.8 with `env -u ANTHROPIC_API_KEY` and read-only repo access. The initial review process stalled after inspecting the launcher/logbook and cross-region paths, so the same session was resumed with tools disabled to get a final verdict.

Verdict:

- No static blockers for the launcher after the import fix.
- The `this_output_path` import fix is correct: `ExecutorStep` and `this_output_path` are defined in `marin.execution.types`, while `ExecutorMainConfig` and `executor_main` remain in `marin.execution.executor`.
- The nonblocking historical metric-source fix is correct: CompletedAdamH metric sources are now `InputName.nonblocking()` and should not schedule old upstream isoflop training steps.
- CC recommends holding live retry submission until the parent-inline analysis/manifest path is validated.

Required before next retry:

1. Run the isoflop analysis/backfill path in isolation and confirm `adamh_scaling_v6` is produced with no W&B `api.run()` failures.
2. Run launcher `--dry-run --analysis-output-path <analysis_output>` and confirm all 8 manifest rows, including the extrapolated `1e21` rung.
3. Confirm the 39 `us-east5` runtime caches for the Dolma3/Dolmino top-level buckets are materialized.
4. Confirm the submitting shell has `MARIN_PREFIX` unset or exactly `gs://marin-us-east5`; the launcher currently uses `os.environ.setdefault`, so a stale inherited prefix would not be overwritten.

Fieldbook updates:

- Added warning validation `cc.launcher_retry_review`.
- Marked unsubmitted retry row `job_01kvw05rhv5tdbj4jxwx61fak8` as failed because no Iris job was created.
- Added next-action note `Before next Delphi baseline retry`.

## 2026-06-24 Final CC Review And Preflight

Patch:

- Replaced local `os.makedirs` / builtin `open` in `save_delphi_baseline_manifest` with fsspec-backed `fs.makedirs` / `fs.open`.
- Built the manifest CSV in `io.StringIO` before writing it through fsspec, so the same path works for `gs://`, local paths, and `memory://`.
- Replaced `os.environ.setdefault("MARIN_PREFIX", ...)` with a hard guard: reject inherited non-east5 `MARIN_PREFIX`, then set `MARIN_PREFIX=gs://marin-us-east5`.

Validation:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes.
- Isolated fsspec smoke under `uv run --no-project` wrote `run_specs.json`, `summary.json`, and `training_manifest.csv` to `memory://`; the CSV had 8 rows from `proportional_3e18` through `unimax8_1e21`.
- Isolated dry-run under `env -u MARIN_PREFIX uv run --no-project ... python -m experiments.domain_phase_mix.launch_delphi_baseline_mixtures --dry-run --analysis-output-path <preflight>` completed and regenerated the 8-row manifest.

Final CC review:

- Claude Code was invoked via `claude -p` with `env -u ANTHROPIC_API_KEY`, Opus 4.8, and read-only repo access.
- Verdict: **GO; no blockers.**
- CC explicitly reviewed and accepted the fsspec manifest patch.
- CC explicitly reviewed and accepted the `MARIN_PREFIX` patch, noting that `rigging.filesystem.marin_prefix()` reads the environment live and `executor_main` falls through to that value because `ExecutorMainConfig.prefix` defaults to `None`.
- CC assessed cross-region/upstream scheduling risk as low: historical CompletedAdamH sources are nonblocking; top-level domains were validated as east5 existing-cache components; no old isoflop training steps should be scheduled.

Submission constraint:

- Keep Iris parent placement on the wrapper: `--region us-east5 --zone us-east5-a`.
- Keep launcher arguments separate: pass `--max-concurrent 4` to the launcher; do not pass Iris `--region/--zone` through to the launcher.

## 2026-06-24 Retry Failed On Analysis Artifact Serialization

Submitted:

- Iris parent: `/calvinxu/dm-delphi-baseline-mixtures-issue6607-20260623-2244`
- Fieldbook job: `job_01kvw2h7h7kwf5fs9sr3xx71z6`

Result:

- Parent started on east5 and executed the isoflop analysis step.
- The analysis step successfully wrote `isoflop_analysis_result.json` and `fit_curves.json` to `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355/`.
- The parent failed before launching child TPU jobs because the executor tried to persist the `FitScalingLawsResult` return value as `.artifact.json`, and that dataclass contains tuple-keyed `fit_curves`.
- Error: `TypeError: keys must be str, int, float, bool or None, not tuple`.

Patch:

- Added `run_delphi_isoflop_analysis_step`, a launcher-local wrapper around `run_isoflop_analysis_step`.
- The wrapper keeps the existing file-writing side effects, then returns `PathMetadata(path=config.output_path)` so executor artifact serialization is JSON-safe.
- The analysis JSON files remain the source of truth consumed by downstream manifest/training resolution.

Validation:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes.
- Minimal artifact smoke under isolated `uv --no-project` confirms the wrapper returns `PathMetadata` and `Artifact.save` writes `.artifact.json`.
- The dry-run manifest path still resolves all 8 rows after the wrapper patch.

## 2026-06-24 Retry Failed On Parent-Side JAX Initialization

Submitted:

- Iris parent: `/calvinxu/dm-delphi-baseline-mixtures-issue6607-20260623-2257`
- Fieldbook job: `job_01kvw3915cr0bm75qmshdydhps`

Result:

- Parent reran the isoflop analysis step successfully and wrote the expected JSON files under `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355/`.
- All eight training steps then failed immediately with `RuntimeError: jax.distributed.initialize() must be called before any JAX calls that might initialise the XLA backend`.
- Diagnosis: the analysis step uses JAX in the same executor worker process before `run_levanter_train_lm` initializes Levanter distributed training. Because `run_levanter_train_lm` runs in the current executor process, parent-side JAX/XLA initialization contaminates subsequent training steps.

Patch:

- `--analysis-output-path` now works in live mode, not only `--dry-run`.
- When `--analysis-output-path` is supplied, the launcher omits the isoflop analysis step from the executor graph and passes the materialized analysis path directly to the manifest and training steps.
- The next retry should use `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355`.

Validation:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes.
- Dry-run with the GCS analysis path resolves all 8 manifest rows.
- `CI=1` live graph build with the GCS analysis path reports exactly 8 training steps and skips executor launch, proving the analysis step is omitted.
- East5 launch safety passes for the planned retry command with `--analysis-output-path`.

CC follow-up review:

- Claude Code was invoked via `claude -p` with `env -u ANTHROPIC_API_KEY`, Opus 4.8, and read-only repo access.
- Verdict: **NO-GO** on the analysis-path-only patch.
- Blocker: training `ExecutorStep`s used plain functions with no `resources=`, so `StepRunner` ran them inline in the parent `ThreadPoolExecutor`. Removing the analysis step avoided one JAX initializer, but the training steps would still collide with each other in the same process and could not request their distinct TPU pod shapes.
- Recommended fix: pass `resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone)` on each training `ExecutorStep`, matching the standard Marin training dispatch pattern.

Patch:

- Added explicit TPU resources to every training `ExecutorStep`.
- Kept the live `--analysis-output-path` support, because it prevents parent-side analysis work and pins a materialized, same-region analysis artifact.

Validation after resource patch:

- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes.
- Dry-run with the GCS analysis path still resolves all 8 manifest rows.
- `CI=1` live graph build still reports exactly 8 training steps.
- Explicit graph assertion confirms all 8 training steps have `TpuConfig` resources: `v5p-8`, `v5p-8`, `v5p-16`, `v5p-16`, `v5p-32`, `v5p-32`, `v5p-64`, `v5p-64`.
- East5 launch safety passes for the planned retry command with `--analysis-output-path`.

CC follow-up after resource patch:

- Claude Code was resumed in the same Opus 4.8 session via `claude -p` with `env -u ANTHROPIC_API_KEY`.
- Verdict: **GO; no blockers.**
- CC verified that `resources=ResourceConfig.with_tpu(...)` flips `StepRunner` into the Fray/Iris job submission branch, so each training step should run on its own TPU pod in a fresh process.
- CC also checked that `materialize()` on pod should be a no-op for the materialized GCS cache components, that cross-region risk remains clean, and that the duplicated inner `TrainLmOnPodConfig.resources` is benign.
- Recommendation: run a single cheap smoke first with `--target-budgets 3e18 --mixtures proportional` before full 8-way fan-out, to validate pod dispatch, cloudpickle/import, and JAX initialization end to end.

## 2026-06-24 Smoke Submission After CC GO

Submitted single-rung smoke:

- Iris parent: `/calvinxu/dm-delphi-baseline-mixtures-issue6607-smoke-20260623-2327`
- Fieldbook job: `job_01kvw5014v1j27604qn0zvn6ct`
- Command filters: `--target-budgets 3e18 --mixtures proportional --max-concurrent 1`
- Analysis input: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355`

Observed state:

- Parent is running on Iris with no parent failure or JAX initialization error.
- Manifest step succeeded and selected `proportional_3e18` with `N=3.58e+08`, `tokens=1.58e+09`.
- Training did not run inline in the parent. The parent created child TPU job `/calvinxu/dm-delphi-baseline-mixtures-issue6607-smoke-20260623-2327/pinlin_calvin_xu-data_mixture-delphi_baseline_mixtures_issue6607_20260623-proportional_3e18_c11ee055-94e34ffb`.
- Child TPU job is pending only on capacity: `Scheduler: Insufficient TPUs (need 4, available 0)` for `tpu_v5p-preemptible_8-us-east5-a`.

Interpretation:

- The CC blocker about missing `ExecutorStep.resources` is fixed at the dispatch level: the training step now routes through Iris/Fray as a child TPU job.
- Do not treat the smoke as fully end-to-end passed until the child starts and clears import/cloudpickle/JAX initialization inside the TPU pod.

## 2026-06-24 CC Review And Live Health Refresh

CC review:

- Invoked Claude Code via `env -u ANTHROPIC_API_KEY claude -p --model claude-opus-4-8 --effort max` with read-only repo access.
- First resumed-session attempt stalled without output and was interrupted; the shorter resumed review returned.
- Verdict: **GO**, with no static launch blockers.
- CC's operational gates were (1) prove the smoke clears in-pod startup and (2) verify Paloma/uncheatable validation caches are east5-local.
- CC also noted non-blocking concerns: split submissions reuse positional `data_seed` values, and multi-host rungs need live verification beyond the single-host smoke.

Live follow-up checks:

- The earlier smoke-pending state is stale. The `proportional_3e18` child is now running in-pod on `v5p-8`, passed first train step, logged training progress past step 900, and saved checkpoints at steps 414 and 909 under `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/.../proportional_3e18-ebc4aa/`.
- `unimax8_3e18` is also running in-pod on `v5p-8`, passed first train step, logged training progress past step 400, and saved a checkpoint at step 427.
- The remaining parent has live child jobs for `proportional_2e19`, `unimax8_2e19`, `proportional_3e20`, and `unimax8_3e20`; these have all cleared startup and are logging training progress/checkpoints.
- The `1e21` children are not expected to appear until current slots free because the remaining parent was submitted with `--max-concurrent 4`.
- Live logs show default validation caches loading from east5 for both Paloma and uncheatable eval, e.g. `gs://marin-us-east5/tokenized/paloma/.../validation` and `gs://marin-us-east5/tokenized/uncheatable_eval/.../validation`.
- Warnings about missing validation splits for train buckets are expected: those buckets are training components, not the default Paloma/uncheatable validation set.

## 2026-06-24 00:02 PT Health Poll

Iris prefix `/calvinxu/dm-delphi-baseline-mixtures-issue6607` currently has six running child training jobs plus three running parents. The only failed jobs under the prefix are historical attempts already covered by the retry chain.

Current active child progress:

- `proportional_3e18`: running on `v5p-8`; latest observed progress `1.20kit/3.01kit`, ETA about `36m`, checkpoints saved under the east5 temporary checkpoint root.
- `unimax8_3e18`: running on `v5p-8`; latest observed progress `698/3007`, ETA about `46m`, checkpoint saved at step 427 under the east5 temporary checkpoint root.
- `proportional_2e19`: running on `v5p-16`; latest observed progress `876/9.90kit`, ETA about `3h09m`, checkpoint save in progress at step 895.
- `unimax8_2e19`: running on `v5p-16`; latest observed progress `879/9.90kit`, ETA about `3h05m`, checkpoint saved at step 416.
- `proportional_3e20`: running on `v5p-32`; latest observed progress `206/23.5kit`, ETA about `23h19m`, checkpoint saved at step 136.
- `unimax8_3e20`: running on `v5p-32`; latest observed progress `222/23.5kit`, ETA about `23h29m`, checkpoint saved at step 135.

No active child logs showed OOM, HBM, traceback, JAX distributed initialization, or fatal error signatures in the recent scan. The two `1e21` rows are not visible yet because the remaining parent was submitted with `--max-concurrent 4`; they should launch after the current `2e19`/`3e20` slots free.

## 2026-06-24 00:06 PT Health Poll

Fieldbook remains accurate for the retry history, but its readiness block still counts historical failed parents and old failed validations. Live Iris state is the source of truth for current execution.

Iris prefix `/calvinxu/dm-delphi-baseline-mixtures-issue6607` currently has six running child training jobs and three running parent jobs. The historical failed parents remain visible under the prefix, but no new failures appeared in this poll.

Current active child progress:

- `proportional_3e18`: running on `v5p-8`; latest observed progress `1.40kit/3.01kit`, ETA about `32m`, checkpoint saved at step 1355.
- `unimax8_3e18`: running on `v5p-8`; latest observed progress `849/3007`, ETA about `44m`, checkpoint saved at step 427.
- `proportional_2e19`: running on `v5p-16`; latest observed progress `970/9.90kit`, ETA about `3h03m`, checkpoint saved at step 895.
- `unimax8_2e19`: running on `v5p-16`; latest observed progress `972/9.90kit`, ETA about `3h03m`, checkpoint saved at step 899.
- `proportional_3e20`: running on `v5p-32`; latest observed progress `257/23.5kit`, ETA about `23h14m`; checkpoint was previously observed at step 136.
- `unimax8_3e20`: running on `v5p-32`; latest observed progress `273/23.5kit`, ETA about `23h43m`; checkpoint was previously observed at step 135.

The recent-log fatal-signature pass found zero lines matching OOM, HBM exhaustion, traceback, JAX distributed initialization, `RuntimeError`, `Exception`, `OwnerDiedError`, or dead-node patterns. The two `1e21` rows remain not visible because the remaining parent is occupying its four child slots with the two `2e19` and two `3e20` jobs; they should appear after the first `2e19` child finishes.

## 2026-06-24 00:17 PT Health Poll

After one normal babysit cadence, Iris still shows six running child training jobs and three running parent jobs. No new child jobs reached a terminal failure state; the two `1e21` rows are still not visible because the remaining parent has all four child slots occupied.

Current active child progress:

- `proportional_3e18`: running on `v5p-8`; latest observed progress `1.94kit/3.01kit`, ETA about `21m`, checkpoint saved at step 1849.
- `unimax8_3e18`: running on `v5p-8`; latest observed progress `1.40kit/3.01kit`, ETA about `32m`, checkpoint saved at step 1374.
- `proportional_2e19`: running on `v5p-16`; latest observed progress `1.53kit/9.90kit`, ETA about `2h51m`, checkpoint saved at step 1324.
- `unimax8_2e19`: running on `v5p-16`; latest observed progress `1.53kit/9.90kit`, ETA about `3h33m`, checkpoint saved at step 1331.
- `proportional_3e20`: running on `v5p-32`; latest observed progress `459/23.5kit`, ETA about `23h07m`, checkpoint save in progress at step 467.
- `unimax8_3e20`: running on `v5p-32`; latest observed progress `456/23.5kit`, ETA about `23h48m`, checkpoint save in progress at step 464.

The stricter fatal-signature scan again found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines.

## 2026-06-24 00:20 PT Health Poll

Iris still shows six running child training jobs. No `1e21` child has launched yet, consistent with the remaining parent using its four child slots.

Current active child progress:

- `proportional_3e18`: `2.05kit/3.01kit`, ETA about `19m`, checkpoint saved at step 1849.
- `unimax8_3e18`: `1.55kit/3.01kit`, ETA about `29m`, checkpoint saved at step 1374.
- `proportional_2e19`: `1.68kit/9.90kit`, ETA about `2h49m`.
- `unimax8_2e19`: `1.68kit/9.90kit`, ETA about `2h56m`.
- `proportional_3e20`: `508/23.5kit`, ETA about `23h00m`, checkpoint saved at step 467.
- `unimax8_3e20`: `505/23.5kit`, ETA about `23h11m`, checkpoint saved at step 464.

Fatal-signature scan again found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines.

## 2026-06-24 00:29 PT Health Poll

Iris still shows six running child training jobs and three running parent jobs under prefix `/calvinxu/dm-delphi-baseline-mixtures-issue6607`. No new active child failures appeared. The two `1e21` children are still not visible, which remains consistent with the `remaining` parent using all four child slots for the two `2e19` and two `3e20` children.

Current active child progress:

- `proportional_3e18`: `2.44kit/3.01kit`, ETA about `11m`, checkpoint saved at step 2305.
- `unimax8_3e18`: `1.95kit/3.01kit`, ETA about `21m`, checkpoint saved at step 1868.
- `proportional_2e19`: `2.05kit/9.90kit`, ETA about `2h43m`, checkpoint saved at step 1805.
- `unimax8_2e19`: `2.05kit/9.90kit`, ETA about `2h40m`, checkpoint saved at step 1813.
- `proportional_3e20`: `641/23.5kit`, ETA about `23h56m`, checkpoint saved at step 632.
- `unimax8_3e20`: `638/23.5kit`, ETA about `23h44m`, checkpoint saved at step 628.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines.

Fieldbook cleanup:

- Archived duplicate failed job `job_01kvvzg1apktm4vmjeycf13asz`, a second no-remote production-priority rejection record that had no retry link and was keeping a stale blocker visible.
- Archived superseded validation `val_01kvw4cb34rgrg9ryqs8pa26f0`, the old CC NO-GO for the missing `ExecutorStep.resources` state that was later patched and re-reviewed.
- Archived superseded validation `val_01kvw0wewdbc58yva1902ajtwn`, the old conditional-GO review whose four required pre-retry checks are now recorded as pass validations.
- Resolved next-action note `note_01kvw0wez7exh5q9h7pw91tgwq` because the requested pre-retry checks and live smoke gates have been completed.
- After cleanup, Fieldbook reports zero active blocking failed jobs. Historical failed jobs remain in recovery-in-progress through the active retry chain.

## 2026-06-24 00:36 PT Health Poll

Iris still shows six running child training jobs and three running parent jobs. No new terminal failures appeared. The two `1e21` children are still not visible; this remains expected until the `remaining` parent frees one of its four child slots.

Current active child progress:

- `proportional_3e18`: `2.79kit/3.01kit`, ETA about `4m`, checkpoint save in progress at step 2799.
- `unimax8_3e18`: `2.26kit/3.01kit`, ETA about `16m`, checkpoint saved at step 1868.
- `proportional_2e19`: `2.34kit/9.90kit`, ETA about `2h35m`, checkpoint saved at step 2247.
- `unimax8_2e19`: `2.34kit/9.90kit`, ETA about `2h33m`, checkpoint saved at step 2261.
- `proportional_3e20`: `743/23.5kit`, ETA about `23h32m`.
- `unimax8_3e20`: `757/23.5kit`, ETA about `22h53m`.

Recent-log fatal-signature scan again found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

Terminal success verification:

- `proportional_3e18` parent `/calvinxu/dm-delphi-baseline-mixtures-issue6607-smoke-20260623-2327` reached `JOB_STATE_SUCCEEDED` with `exit_code=0`, `failure_count=0`, and `preemption_count=0`.
- Final Levanter checkpoint exists at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_3e18-ebc4aa/checkpoints/step-3006` and contains `metadata.json`.
- HF export exists at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_3e18-ebc4aa/hf/step-3006` and contains expected config/model/tokenizer files.

## 2026-06-24 00:56 PT Health Poll

`3e18` terminal status:

- `proportional_3e18`: parent and child are `JOB_STATE_SUCCEEDED`; final checkpoint and HF export verified.
- `unimax8_3e18`: parent and child are `JOB_STATE_SUCCEEDED` with `exit_code=0`, `failure_count=0`, `preemption_count=0`; final Levanter checkpoint exists at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/unimax8_3e18-cb3b49/checkpoints/step-3006` with `metadata.json`; HF export exists at `.../unimax8_3e18-cb3b49/hf/step-3006` with config/model/tokenizer files.

Remaining live child progress under parent `/calvinxu/dm-delphi-baseline-mixtures-issue6607-remaining-20260623-2337`:

- `proportional_2e19`: `3.29kit/9.90kit`, ETA about `2h17m`, checkpoint saved at step 3166.
- `unimax8_2e19`: `3.34kit/9.90kit`, ETA about `2h16m`, checkpoint saved at step 3190.
- `proportional_3e20`: `1.07kit/23.5kit`, ETA about `26h56m`, eval completed `63/63`.
- `unimax8_3e20`: `1.07kit/23.5kit`, ETA about `25h37m`, eval completed `63/63`.

The `1e21` children are still not visible because the remaining parent has `--max-concurrent 4` and all four slots are occupied by the `2e19` and `3e20` children. Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 00:42 PT Health Poll

Iris still shows six running child training jobs and three running parent jobs. No new terminal failures appeared. The two `1e21` children are still not visible; this is expected because they belong to the `remaining` parent, whose four child slots are still occupied by the two `2e19` and two `3e20` children. The separate `3e18` parents finishing will not free slots for the `1e21` children.

Current active child progress:

- `proportional_3e18`: reached final eval and checkpoint; saved `step-3006` under `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_3e18-ebc4aa/checkpoints/step-3006`; started HF export to `.../proportional_3e18-ebc4aa/hf/step-3006`. Iris still marks it running, likely final export/cleanup.
- `unimax8_3e18`: `2.56kit/3.01kit`, ETA about `9m`, checkpoint saved at step 2330.
- `proportional_2e19`: `2.53kit/9.90kit`, ETA about `2h30m`, checkpoint saved at step 2247.
- `unimax8_2e19`: `2.53kit/9.90kit`, ETA about `2h30m`, checkpoint saved at step 2261.
- `proportional_3e20`: `808/23.5kit`, ETA about `23h15m`, checkpoint saved at step 797.
- `unimax8_3e20`: `804/23.5kit`, ETA about `25h44m`, checkpoint saved at step 792.

Recent-log fatal-signature scan again found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 CC Review Of Current Launcher Plan

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p` in the resumed Marin data-mixing review session with Opus 4.8, max effort, and read-only access to the repo.

Verdict: **GO; no blockers.**

CC specifically cleared:

- The remaining-rung in-pod path does not reintroduce the earlier JAX distributed-initialization bug. The child pre-train path reads scaling fits and calls `predict_optimal_config`, which CC verified is pure Python; the JAX-using fitting work remains outside the child training process when launching with the precomputed `--analysis-output-path`.
- Mixture and scale selection are internally consistent: proportional and UniMax-8 over 39 domains, identical `phase_0`/`phase_1`, `2e19 -> v5p-16`, `3e20 -> v5p-32`, and `1e21 -> v5p-64`.
- Cross-region risk is clean: launcher guards `MARIN_PREFIX`, pins rows to `us-east5/us-east5-a`, uses `gs://marin-us-east5`, and keeps historical CompletedAdamH sources nonblocking.
- Child dispatch is correct: every training `ExecutorStep` has TPU `resources=...`, so `StepRunner` uses the Iris child-job branch instead of running training inline in parent threads.
- The simulated-epoching convention is scale-invariant: with fixed `target_budget=TARGET_BUDGET_DOLMA3_COMMON_CRAWL` and per-rung `experiment_budget=realized_train_tokens`, materialized epochs per bucket are \(w_i \cdot \mathrm{target\_budget}/|G_i|\), independent of the scaling rung.

Important non-blockers to preserve:

- Retry commands must keep the same argument order/membership because `run_id`/`data_seed` are positional and contribute to step hashes.
- Launch children with the same precomputed analysis path used for the reviewed manifest.
- `1e21` is a scientific/operational extrapolation risk, not a static launcher blocker; monitor first compile, OOM/HBM signatures, and the loss curve.
- Multi-host rungs still require live startup/progress checks despite the single-host smoke passing.

## 2026-06-24 01:07 PT Health Poll

Iris prefix `/calvinxu/dm-delphi-baseline-mixtures-issue6607` currently shows:

- Parents: one running parent (`remaining`) and two succeeded parents (`proportional_3e18`, `unimax8_3e18`); historical failed parents remain recovered in the retry chain.
- Children: two succeeded children (`proportional_3e18`, `unimax8_3e18`) and four running children (`proportional_2e19`, `unimax8_2e19`, `proportional_3e20`, `unimax8_3e20`).
- The `1e21` children are still not visible because the `remaining` parent has `--max-concurrent 4` and all four slots are occupied by the two `2e19` and two `3e20` children.

Completed artifact verification:

- `proportional_3e18`: final checkpoint metadata and HF export config exist under `gs://marin-us-east5/.../proportional_3e18-ebc4aa/.../step-3006`.
- `unimax8_3e18`: final checkpoint metadata and HF export config exist under `gs://marin-us-east5/.../unimax8_3e18-cb3b49/.../step-3006`.

Current active child progress:

- `proportional_2e19`: `3.87kit/9.90kit`, ETA about `2h03m`, checkpoint saved at step `3647`.
- `unimax8_2e19`: `3.87kit/9.90kit`, ETA about `2h06m`.
- `proportional_3e20`: `1.25kit/23.5kit`, ETA about `22h55m`, checkpoint saved at step `1107`.
- `unimax8_3e20`: `1.25kit/23.5kit`, ETA about `22h32m`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:19 PT Health Poll

Follow-up cadence after the 01:07 poll:

- Iris job structure is unchanged: two succeeded children (`proportional_3e18`, `unimax8_3e18`), four running children (`proportional_2e19`, `unimax8_2e19`, `proportional_3e20`, `unimax8_3e20`), and the `1e21` children still not visible because the `remaining` parent has all four slots occupied.
- Fieldbook still has zero active blocking failed jobs. The remaining warning is the known local `uv` validation warning, not a live execution blocker.

Current active child progress:

- `proportional_2e19`: `4.38kit/9.90kit`, ETA about `1h53m`, checkpoint saved at step `4091`.
- `unimax8_2e19`: `4.39kit/9.90kit`, ETA about `1h52m`, checkpoint saved at step `4120`.
- `proportional_3e20`: `1.46kit/23.5kit`, ETA about `22h10m`, checkpoint saved at step `1437`.
- `unimax8_3e20`: `1.45kit/23.5kit`, ETA about `22h20m`, checkpoint saved at step `1427`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll. Next important transition is when either `2e19` child finishes and frees a `remaining` parent slot, at which point one of the `1e21` children should appear and clear startup.

## 2026-06-24 01:22 PT Health Poll

Iris structure remains stable: two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `4.53kit/9.90kit`, ETA about `1h50m`, checkpoint saved at step `4091`.
- `unimax8_2e19`: `4.54kit/9.90kit`, ETA about `1h51m`, checkpoint saved at step `4120`.
- `proportional_3e20`: `1.51kit/23.5kit`, ETA about `22h02m`, checkpoint saved at step `1437`.
- `unimax8_3e20`: `1.49kit/23.5kit`, ETA about `22h28m`, checkpoint saved at step `1427`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:25 PT Health Poll

Iris structure remains stable: two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `4.67kit/9.90kit`, ETA about `1h51m`, checkpoint saved at step `4571`.
- `unimax8_2e19`: `4.68kit/9.90kit`, ETA about `1h47m`, checkpoint saved at step `4601`.
- `proportional_3e20`: `1.54kit/23.5kit`, ETA about `22h36m`, checkpoint saved at step `1437`.
- `unimax8_3e20`: `1.54kit/23.5kit`, ETA about `22h12m`, checkpoint saved at step `1427`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:36 PT Health Poll

After one normal babysit cadence, Iris still shows two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `5.19kit/9.90kit`, ETA about `1h36m`, checkpoint saved at step `5000`.
- `unimax8_2e19`: `5.20kit/9.90kit`, ETA about `1h35m`, checkpoint saved at step `5000`.
- `proportional_3e20`: `1.72kit/23.5kit`, ETA about `25h05m`, checkpoint saved at step `1601`.
- `unimax8_3e20`: `1.72kit/23.5kit`, ETA about `22h20m`, checkpoint saved at step `1591`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:48 PT Health Poll

After another babysit cadence, Iris still shows two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `5.72kit/9.90kit`, ETA about `1h25m`, checkpoint saved at step `5442`.
- `unimax8_2e19`: `5.78kit/9.90kit`, ETA about `1h24m`, checkpoint saved at step `5447`.
- `proportional_3e20`: `1.93kit/23.5kit`, ETA about `22h30m`, checkpoint saved at step `1766`.
- `unimax8_3e20`: `1.92kit/23.5kit`, ETA about `35h57m`, checkpoint saved at step `1755`. The ETA is noisier here than prior polls, but progress and checkpoints are still advancing and there are no fatal signatures.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:51 PT Health Poll

Iris structure remains stable: two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `5.87kit/9.90kit`, ETA about `1h24m`, checkpoint saved at step `5442`.
- `unimax8_2e19`: `5.92kit/9.90kit`, ETA about `1h21m`, checkpoint saved at step `5447`.
- `proportional_3e20`: `1.98kit/23.5kit`, ETA about `21h34m`, checkpoint saved at step `1932`.
- `unimax8_3e20`: `1.97kit/23.5kit`, ETA about `21h44m`, checkpoint saved at step `1920`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 01:54 PT Health Poll

Iris structure remains stable: two succeeded children, four running children, one running `remaining` parent, and no visible `1e21` children yet because the `remaining` parent still has all four `--max-concurrent` slots occupied.

Current active child progress:

- `proportional_2e19`: `5.96kit/9.90kit`, ETA about `1h26m`, checkpoint saved at step `5924`.
- `unimax8_2e19`: `6.00kit/9.90kit`, checkpoint saved at step `5929`. The recent ETA line is noisy/transient (`16h19m`) and should not be treated as a regression unless it persists across cadence checks.
- `proportional_3e20`: `2.00kit/23.5kit`, checkpoint saved at step `1932`. The recent ETA line is noisy/transient (`144h02m`) and should not be treated as a regression unless it persists across cadence checks.
- `unimax8_3e20`: `1.99kit/23.5kit`, ETA about `21h39m`, checkpoint saved at step `1920`.

Recent-log fatal-signature scan found zero OOM, HBM, traceback, JAX distributed initialization, runtime-exception, owner-death, or dead-node lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 02:54 PT CC Review And Health Poll

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, and read-only tools. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **GO**. CC found no blockers requiring stop, resubmit, or patch before the `1e21` children launch. CC specifically re-cleared child dispatch, parent-side JAX-init avoidance with the precomputed analysis path, east5 locality, mixture/scale coherence, invariant simulated epoching, and the training/eval split. The remaining caveat is operational: when `1e21` starts, verify first XLA compile, first checkpoint, finite/decreasing loss, and no OOM/HBM signatures.

Read-only checks after review:

- Current remaining-manifest rows are `2e19`, `3e20`, and `1e21` for both proportional and UniMax-8, all `us-east5/us-east5-a`, with expected TPU types `v5p-16`, `v5p-32`, and `v5p-64`.
- Manifest phase weights cover 39 domains, sum to 1.0 in both phases, and have `phase_0 == phase_1` for every row.
- Iris summaries show four live children, zero failures, zero preemptions, and expected east5 workers: two `v5p-16` jobs for `2e19` and two `v5p-32` jobs for `3e20`.
- Tail logs show active progress/checkpointing: `proportional_2e19` around `8.72kit/9.90kit`, `unimax8_2e19` around `8.73kit/9.90kit`, and both `3e20` runs around `2.96kit/23.5kit`.
- Recent fatal-signature grep over the remaining parent found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, or preemption lines.
- Verified final `3e18` checkpoint metadata and HF config markers exist for both proportional and UniMax-8 under `gs://marin-us-east5/.../step-3006`.

Expected near-term transition: the two `2e19` children should finish in roughly 25-40 minutes based on tail-log ETA, freeing slots for `1e21`. The next babysit poll should confirm that the `1e21` children appear and clear startup.

## 2026-06-24 03:17 PT CC Launcher Review

Claude Code was invoked again via `env -u ANTHROPIC_API_KEY claude`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, and read-only tools. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **GO**. CC found no code patch required before further launch or retry decisions.

The most important clarification is that omitting `--analysis-output-path` still creates an inline parent analysis step that can initialize JAX/XLA, but this is no longer fatal because training steps now child-dispatch to fresh TPU pods. Even so, retries should treat the precomputed `--analysis-output-path` as mandatory to avoid analysis recompute, W&B-read risk, and analysis drift.

The other key operational rule is retry stability: same-command retries with identical `--target-budgets` and `--mixtures` values and order are safe, but narrowed or reordered retries can shift positional `run_id` / `data_seed`, changing step config hashes and forking output paths. Avoid narrowed retries for this launcher unless the seed mapping is later made stable by `(mixture, target_flops)`.

CC re-cleared east5 locality, child TPU dispatch, `MARIN_PREFIX`, proportional and UniMax-8 phase weights, invariant simulated-epoch exposure, and `max_concurrent=4`. Remaining monitor-only caveat: watch the `1e21` children when slots free for first XLA compile, first checkpoint, finite/decreasing loss, HBM headroom, and OOM/preempt-without-progress signatures.

## 2026-06-24 03:25 PT Health Poll

The `2e19` pair has now terminal-succeeded:

- `proportional_2e19`: child Iris state `JOB_STATE_SUCCEEDED`; final Levanter checkpoint metadata and HF config verified at `step-9901`.
- `unimax8_2e19`: child Iris state `JOB_STATE_SUCCEEDED`; final Levanter checkpoint metadata and HF config verified at `step-9901`.

The `3e20` pair remains healthy and advancing:

- `proportional_3e20`: around `3.44kit/23.5kit`, rate about `3.6s/it`, ETA about `20h07m`, recent loss around `3.04`; fresh temporary checkpoint at step `3381`.
- `unimax8_3e20`: around `3.44kit/23.5kit`, rate about `3.6s/it`, ETA about `20h16m`, recent loss around `2.87`; fresh temporary checkpoint at step `3361`.

Both `1e21` children have been submitted:

- `proportional_1e21`: `JOB_STATE_PENDING`, 8 pending tasks, zero failures, zero preemptions.
- `unimax8_1e21`: `JOB_STATE_PENDING`, 8 pending tasks, zero failures, zero preemptions.

The pending reason for both `1e21` children is scheduler capacity only: coscheduling needs 8 workers / 4 TPUs, available 0, with autoscaler `tier_blocked` quota-pool tier monotonicity. This is not an application failure and does not warrant a retry. Recent-log fatal-signature scan found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, or preemption lines.

Next check: wait for v5p-64 allocation. Once either `1e21` child starts, verify first XLA compile, first train progress, finite/decreasing loss, first checkpoint, HBM headroom, and no OOM/preempt-without-progress signatures.

## 2026-06-24 03:28 PT Health Poll

CC review remains current: the latest Opus 4.8 review returned **GO** with no launcher patch required. The relevant caveats remain retry discipline, using the precomputed analysis output path, and watching the `1e21` children once v5p-64 capacity appears.

Iris state remains healthy:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` are terminal-succeeded.
- `proportional_2e19` final Levanter checkpoint metadata and HF config were re-verified at `step-9901`.
- `unimax8_2e19` final Levanter checkpoint metadata and HF config were re-verified at `step-9901`.
- `proportional_3e20` is still running on four tasks, around `3.51kit/23.5kit`, recent rate about `3.6s/it`, ETA about `20h02m`, recent loss around `3.11`.
- `unimax8_3e20` is still running on four tasks, around `3.49kit/23.5kit`, recent rate about `3.6s/it`, ETA about `20h09m`, recent loss around `2.82`.
- `proportional_1e21` and `unimax8_1e21` remain `JOB_STATE_PENDING` with 8 pending tasks each, zero failures, and zero preemptions.

The `1e21` pending reason is unchanged scheduler capacity: each child needs 8 workers / 4 TPUs and current v5p-64 capacity is unavailable, with autoscaler `tier_blocked` quota-pool tier monotonicity. This is not an application failure and does not warrant retry.

Recent-log fatal-signature scan found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, preemption, or resource-exhaustion lines.

## 2026-06-24 03:30 PT Health Poll

Fieldbook remains aligned with the current Iris state: no blocking failed jobs; prior failed parents are recovered historical attempts.

Iris state:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` are terminal-succeeded.
- `proportional_3e20` is running on four tasks, around `3.54kit/23.5kit`, recent rate about `3.6s/it`, ETA about `19h59m`, recent loss around `3.06`; temporary checkpoint saved at step `3546`.
- `unimax8_3e20` is running on four tasks, around `3.52kit/23.5kit`, recent rate about `3.6s/it`, ETA about `20h07m`, recent loss around `2.78`; temporary checkpoint saved at step `3526`.
- `proportional_1e21` and `unimax8_1e21` remain `JOB_STATE_PENDING` with 8 pending tasks each, zero failures, and zero preemptions.

The `1e21` pending reason is unchanged and capacity-only: coscheduling needs 8 workers / 4 TPUs, available 0, with autoscaler `tier_blocked` quota-pool tier monotonicity. No retry is indicated.

Recent-log fatal-signature scan found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, preemption, resource-exhaustion, killed, or nonzero exit-code lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 03:44 PT CC Launcher Review

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, and read-only tools. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **GO**. CC found no blockers that should prevent submission of the planned Delphi baseline launcher command, provided the exact final command is validated before submit.

Cleared items:

- Training steps child-dispatch to fresh TPU pods via explicit `ResourceConfig.with_tpu(...)`; the launcher does not inline train on the parent.
- With a supplied `--analysis-output-path`, the parent analysis step is omitted; without it, analysis recompute is still a retry/drift risk even though child dispatch prevents parent JAX initialization from contaminating training children.
- East5 locality is covered by both launcher guards and the command validator: parent `--region us-east5 --zone us-east5-a`, child `MARIN_PREFIX=gs://marin-us-east5`, and command-level `gs://marin-us-*` path checks.
- Proportional and UniMax-8 mixtures have valid identical weights in both phases, and the fixed `target_budget` / rung-specific `experiment_budget` convention keeps materialized epoch exposure invariant across scaling rungs.

Important caveats:

- Retry hash stability depends on the exact CLI arguments, the exact `--analysis-output-path`, and the same code state. Reordering or narrowing `--target-budgets` / `--mixtures`, changing validation configs, or changing launcher constants can fork output paths.
- W&B run names are auto-generated; provenance remains recoverable through tags and output paths, but run names should not be treated as canonical.
- The `1e21` rung remains monitor-only risk: v5p-64 capacity is the likely gate, and once it starts we must verify first XLA compile, first train progress, finite/decreasing loss, first checkpoint, and no OOM/HBM/preempt-without-progress signatures.

## 2026-06-24 03:56 PT Health Poll

Important transition: `proportional_1e21` has started on v5p-64 and cleared the startup checks that CC asked us to watch.

Iris state:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` are terminal-succeeded.
- `proportional_3e20` remains running on four tasks, around `3.96kit/23.5kit`, with recent rate about `3.8s/it`, ETA about `20h43m`, and loss around `3.08`.
- `unimax8_3e20` remains running on four tasks, around `3.94kit/23.5kit`, with recent rate about `3.9s/it`, ETA about `21h14m`, and loss around `2.78`.
- `proportional_1e21` is running on all eight tasks, cleared first train step, reached about `120it/22.1kit`, has recent rate about `7.1s/it`, ETA about `43h`, and loss decreased from about `11.8` at 12 steps to about `7.46` at 120 steps.
- `proportional_1e21` wrote its first temporary checkpoint at `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/proportional_1e21-2f1a48/checkpoints/step-74`.
- `unimax8_1e21` remains `JOB_STATE_PENDING` with 8 pending tasks, zero failures, and zero preemptions.

The `unimax8_1e21` pending reason is capacity-only: coscheduling needs 8 workers / 4 TPUs, available 0, with autoscaler `tier_blocked` quota-pool tier monotonicity. This is not an application failure and does not warrant retry.

Recent-log fatal-signature scan found zero real OOM, out-of-memory, HBM, traceback, runtime-exception, dead-node, owner-death, preemption, resource-exhaustion, killed, or nonzero exit-code lines. One grep hit was a W&B dummy run id containing the substring `hbm`; it is not an HBM warning.

Next check: continue normal cadence. The remaining unverified health gate is `unimax8_1e21` startup once v5p-64 capacity appears.

## 2026-06-24 04:02 PT CC Launcher Review

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, neutral cwd `/tmp/codex-cc-neutral`, read-only tools, and `--add-dir /Users/calvinxu/Projects/Work/Marin/marin`. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **GO**. CC found no launch blockers in `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py`.

Cleared items:

- Every training step uses `ResourceConfig.with_tpu(...)`, so training child-dispatches to TPU pods instead of running on the CPU parent.
- Passing `--analysis-output-path` avoids the inline parent analysis step, and child-side config prediction remains pure Python before Levanter distributed initialization.
- Proportional and UniMax-8 phase weights are valid, cover the 39 top-level domains, and use identical `phase_0` / `phase_1` weights.
- The fixed `target_budget` plus rung-specific `experiment_budget` keeps materialized per-bucket epoch exposure invariant across rungs.
- East5 locality is guarded by the launcher and command-level safety checks: parent region/zone, child `MARIN_PREFIX`, and `gs://marin-us-east5` paths.

Important caveats:

- Do not run an overlapping second parent while the current one is healthy; a second parent would contend on the same executor step locks rather than cleanly skip.
- Retry hash stability requires the same commit, exact `--analysis-output-path`, and byte-identical `--mixtures` / `--target-budgets` order and values. Narrowing or reordering can fork positional `run_id` / `data_seed` and output hashes.
- A wrong analysis path should fail fast if it lacks `adamh_scaling_v6`, but still would waste a submission; verify the analysis JSON before any new retry.
- W&B run names are auto-generated, so provenance should use tags and output paths rather than names.
- The remaining operational risk is still `1e21`, especially `unimax8_1e21` startup when v5p-64 capacity appears.

Recommended checks before any new submit/retry: validate the exact east5 command, verify `isoflop_analysis_result.json["scaling_fits"]["adamh_scaling_v6"]`, dry-run with the same `--analysis-output-path`, confirm the prior parent is terminated before retry, and keep `MARIN_PREFIX` unset or `gs://marin-us-east5`.

## 2026-06-24 04:19 PT CC Launcher Review

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, neutral cwd `/tmp/codex-cc-neutral`, read-only tools, and `--add-dir /Users/calvinxu/Projects/Work/Marin/marin`. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **GO**. CC found no launcher blockers and no reason to patch, stop, or resubmit the current parent.

Cleared items:

- Child dispatch is confirmed in both code and live state: smaller rungs have succeeded, `3e20` runs are progressing, and `proportional_1e21` has started as a child TPU pod rather than inline on the parent.
- `--analysis-output-path` omits the parent-side analysis step; remaining child-side config prediction is pure Python before Levanter distributed initialization.
- East5 locality and `MARIN_PREFIX` guards remain intact.
- Proportional and UniMax-8 mixtures remain valid 39-domain, two-phase-identical mixtures; fixed `target_budget` plus rung-specific `experiment_budget` keeps materialized per-bucket exposure invariant across rungs.
- Same-argument retries are output-hash stable.

Important caveats:

- `unimax8_1e21` is still capacity-gated only; do not retry just because it is pending for v5p-64 capacity.
- Continue monitoring `proportional_1e21` for finite/decreasing loss, HBM/OOM signatures, checkpointing, and preempt-without-progress.
- If a retry is ever needed, terminate the current parent first and use the same commit, exact `--analysis-output-path`, and byte-identical `--mixtures` / `--target-budgets` order and values.
- Downstream HF export and DCLM Core v2 evals are separate from this training launcher and still need to be scheduled after final checkpoints are ready.

## 2026-06-24 03:35 PT Health Poll

The requested CC launcher review is already recorded and remains current: latest Opus 4.8 verdict was **GO**, with no launcher patch required. No additional CC invocation is needed unless a new failure or resubmission decision appears.

Iris state remains healthy:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` are terminal-succeeded.
- `proportional_3e20` is still running on four tasks, around `3.62kit/23.5kit`, recent rate about `3.7s/it`, recent loss around `3.01`; latest verified temporary checkpoint is step `3546`.
- `unimax8_3e20` is still running on four tasks, around `3.60kit/23.5kit`, recent rate about `3.6s/it`, recent loss around `2.87`; latest verified temporary checkpoint is step `3526`.
- `proportional_1e21` and `unimax8_1e21` remain `JOB_STATE_PENDING` with 8 pending tasks each, zero failures, and zero preemptions.

The `1e21` pending reason is capacity-only and slightly improved relative to earlier quota-tier wording: the scheduler needs 8 workers / 4 TPUs and the autoscaler is waiting for `tpu_v5p-preemptible_64-us-east5-a` workers to become ready, selected via demand-routed capacity. This is not an application failure and does not warrant retry.

Recent-log fatal-signature scan found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, preemption, resource-exhaustion, killed, or nonzero exit-code lines. Next important transition remains `1e21` startup: verify first XLA compile, first train progress, finite/decreasing loss, first checkpoint, and no OOM/HBM signatures.

## 2026-06-24 03:37 PT Health Poll

Iris state remains healthy:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` remain terminal-succeeded.
- `proportional_3e20` is running on four tasks, around `3.66kit/23.5kit`, recent rate about `3.7s/it`, ETA about `20h13m`, recent loss around `3.03`. Latest verified temporary checkpoint remains step `3546`.
- `unimax8_3e20` is running on four tasks, around `3.64kit/23.5kit`, recent rate about `3.6s/it`, ETA about `20h01m`, recent loss around `2.88`. Latest verified temporary checkpoint remains step `3526`.
- `proportional_1e21` and `unimax8_1e21` remain `JOB_STATE_PENDING` with 8 pending tasks each, zero failures, and zero preemptions.

The `1e21` pending reason is capacity-only: coscheduling needs 8 workers / 4 TPUs, available 0, with autoscaler `tier_blocked` quota-pool tier monotonicity. This continues to be a scheduler capacity wait, not an application failure.

Recent-log fatal-signature scan found zero OOM, out-of-memory, HBM, traceback, runtime-exception, JAX distributed-init, dead-node, owner-death, preemption, resource-exhaustion, killed, or nonzero exit-code lines. No CC review or resubmission is needed from this poll.

## 2026-06-24 09:15 PT CC Launcher Review

Claude Code was invoked via `env -u ANTHROPIC_API_KEY claude -p`, resumed Marin session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`, Opus 4.8, max effort, neutral cwd `/tmp/codex-cc-neutral`, read-only tools, and `--add-dir /Users/calvinxu/Projects/Work/Marin/marin`. OAuth preflight showed `plambdafour@proton.me` with subscription billing and no inherited `ANTHROPIC_API_KEY`.

Verdict: **CONDITIONAL GO**. CC found no code-level blockers in `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py`. The condition is limited to standard pre-submit checks: exact command passes `east5_launch_safety`, the supplied analysis JSON contains `scaling_fits["adamh_scaling_v6"]`, and east5 validation caches exist. These are not code changes.

Cleared items:

- Passing `--analysis-output-path` omits the parent-side analysis step and avoids the previous parent-side JAX/XLA initialization failure mode.
- Each training step has TPU resources and dispatches to child pods; the parent only runs the JAX-free manifest step.
- East5 locality is guarded by region/zone checks, `MARIN_PREFIX=gs://marin-us-east5`, and `gs://marin-us-east5` output paths.
- Proportional and UniMax-8 weights are nonnegative, normalized, and identical across `phase_0` / `phase_1`.
- Fixed `target_budget` with rung-specific `experiment_budget` keeps simulated-epoch exposure invariant across rungs.
- The launcher does not schedule DCLM Core v2, OLMoBaseEval, or other downstream evals; those remain separate follow-up jobs after training.

Important caveats:

- Retry stability is positional: same commit, same `--analysis-output-path`, and byte-identical mixture / target-budget order are required to preserve output hashes and avoid forking `run_id` / `data_seed`.
- A wrong analysis path should fail fast rather than silently corrupting the run, but still wastes a submission.
- HF export and downstream evals are not configured in this launcher.
- W&B provenance should use tags and output paths; run names are auto-generated.

## 2026-06-24 09:18 PT Health Poll

Fresh Iris poll shows the parent and all four remaining training children running. The `3e18` and `2e19` pairs are already terminal-succeeded from earlier polls. Current live children:

- `proportional_3e20`: running, zero failures, zero preemptions; latest progress `8.92kit/23.5kit`, rate about `3.6s/it`, remaining about `14h41m`, loss about `2.84`; latest observed east5 checkpoint step `8859`.
- `unimax8_3e20`: running, zero failures, zero preemptions; latest progress `8.92kit/23.5kit`, rate about `3.6s/it`, remaining about `14h37m`, loss about `2.60`; latest observed east5 checkpoint step `8853`.
- `proportional_1e21`: running, zero failures, one preemption/coscheduling bounce; replacement workers are running, latest progress `2.64kit/22.1kit`, rate about `7.0s/it`, remaining about `37h32m`, loss about `3.01`; latest observed east5 checkpoint step `2644`.
- `unimax8_1e21`: running, zero failures, zero preemptions; latest progress `2.44kit/22.1kit`, rate about `6.8s/it`, remaining about `37h17m`, loss about `2.71`; latest observed east5 checkpoint step `2406`.

The previous `proportional_1e21` distributed-service fatal line appears tied to preemptible worker/coscheduling bounce and has recovered: the replacement child has finite/decreasing loss, progress, and checkpoints. Recent attention scan found no OOM, out-of-memory, HBM, `RESOURCE_EXHAUSTED`, traceback, runtime exception, killed process, nonzero exit code, or application failure signatures in current progress. No patch, resubmission, or additional CC review is needed from this poll.

## 2026-06-24 09:23 PT Health Poll

Fresh Iris state remains healthy:

- Parent `/calvinxu/dm-delphi-baseline-mixtures-issue6607-remaining-20260623-2337` is still `JOB_STATE_RUNNING`, with zero failures and one known parent-level preemption.
- `proportional_3e20` is running on four tasks, zero failures, zero preemptions. Latest progress: `8.99kit/23.5kit`, rate about `3.7s/it`, remaining about `15h06m`, loss about `2.85`; latest observed east5 checkpoint step `8859`.
- `unimax8_3e20` is running on four tasks, zero failures, zero preemptions. Latest progress: `8.99kit/23.5kit`, rate about `3.6s/it`, remaining about `14h33m`, loss about `2.66`; latest observed east5 checkpoint step `8853`.
- `proportional_1e21` is running on eight tasks, zero failures, one preemptible/coscheduling bounce. Latest progress: `2.69kit/22.1kit`, rate about `6.8s/it`, remaining about `36h35m`, loss about `2.98`; latest observed east5 checkpoint step `2644`.
- `unimax8_1e21` is running on eight tasks, zero failures, zero preemptions. Latest progress: `2.48kit/22.1kit`, rate about `6.8s/it`, remaining about `37h08m`, loss about `2.76`; latest observed east5 checkpoint step `2406`.

The recent-log fatal scan found zero OOM, out-of-memory, HBM, `RESOURCE_EXHAUSTED`, traceback, runtime exception, killed-process, nonzero-exit, or application-failure signatures. The only attention lines are still the recovered `proportional_1e21` distributed-service fatal from the earlier preemptible/coscheduling bounce. No patch, resubmission, or CC review is needed from this poll.

## 2026-06-24 09:28 PT Health Poll

Fresh Iris state remains healthy:

- Parent `/calvinxu/dm-delphi-baseline-mixtures-issue6607-remaining-20260623-2337` remains `JOB_STATE_RUNNING`, with zero failures and one known parent-level preemption.
- `proportional_3e20` is running on four tasks, zero failures, zero preemptions. Latest progress: `9.07kit/23.5kit`, rate about `3.6s/it`, remaining about `14h34m`, loss about `2.89`; latest observed east5 checkpoint step `9004`.
- `unimax8_3e20` is running on four tasks, zero failures, zero preemptions. Latest progress: `9.05kit/23.5kit`, rate about `3.7s/it`, remaining about `14h42m`, loss about `2.67`; latest observed east5 checkpoint step `9001`.
- `proportional_1e21` is running on eight tasks, zero failures, one known preemptible/coscheduling bounce. Latest progress: `2.73kit/22.1kit`, rate about `7.0s/it`, remaining about `37h45m`, loss about `2.97`; latest observed east5 checkpoint step `2644`.
- `unimax8_1e21` is running on eight tasks, zero failures, zero preemptions. Latest progress: `2.52kit/22.1kit`, rate about `7.0s/it`, remaining about `38h06m`, loss about `2.79`; latest observed east5 checkpoint step `2494`.

The recent-log fatal scan again found zero OOM, out-of-memory, HBM, `RESOURCE_EXHAUSTED`, traceback, runtime exception, killed-process, nonzero-exit, or application-failure signatures. The only attention lines remain the already-known recovered `proportional_1e21` distributed-service fatal from the earlier preemptible/coscheduling bounce. No patch, resubmission, or CC review is needed from this poll.

## 2026-06-24 10:22 PT Split CC Launcher Review

Ran an additional Claude Code review with `env -u ANTHROPIC_API_KEY claude --model claude-opus-4-8 --effort max -p <split compact prompts>`. OAuth preflight remained subscription-safe: account `plambdafour@proton.me`, subscription billing, and no inherited `ANTHROPIC_API_KEY`.

The initial read-only tool prompts stalled, so the review was split into compact no-tool prompts. CC raised two apparent blockers: a possible hardcoded `v5p-8` in `_build_mixture_data`, and a possible missing analysis-to-training dependency edge. Local verification resolved both as false positives against the actual launcher:

- Actual training pod resources use `config.tpu_type`, `config.tpu_region`, and `config.tpu_zone` per rung.
- When analysis output is not supplied, `analysis_step.as_input_name()` is assigned before `build_launch_artifacts`, so training configs receive the analysis-step output reference.
- Static checks still pass for 39-domain coverage, proportional normalization, and east5 runtime-cache default.
- `python -m py_compile experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` passes; `uv run python -m py_compile ...` remains blocked by the unrelated workspace dependency-resolution conflict between `tpu-inference` and CPU `vllm` `numba` pins.

Fieldbook validation recorded as `val_01kvxaasheejqge82w62c58rvh` under check `cc.delphi_launcher_split_review_20260624_1018`. Verdict: **no new launcher patch required**.

## 2026-06-24 10:28 PT Health Poll

Fresh Iris poll confirms the two baseline mixtures are running on the intended four-rung ladder: `3e18` on v5p-8, `2e19` on v5p-16, `3e20` on v5p-32, and `1e21` on v5p-64. We are not waiting on v5p-128.

Current state:

- `proportional_3e18`, `unimax8_3e18`, `proportional_2e19`, and `unimax8_2e19` are terminal-succeeded.
- `proportional_3e20` is running on four v5p-32 tasks, zero failures, zero preemptions. Latest progress is about `10.0kit/23.5kit`, recent rate about `3.8s/it`, remaining about `14h09m`, recent loss around `2.85`.
- `unimax8_3e20` is running on four v5p-32 tasks, zero failures, zero preemptions. Latest progress is about `10.0kit/23.5kit`, recent rate about `3.7s/it`, remaining about `13h59m`, recent loss around `2.66`.
- `proportional_1e21` is running on eight v5p-64 tasks, zero failures, one known preemptible/coscheduling bounce that recovered. Latest progress is about `3.23kit/22.1kit`, recent rate about `6.8s/it`, remaining about `35h42m`, recent loss around `2.92`.
- `unimax8_1e21` is running on eight v5p-64 tasks, zero failures, zero preemptions. Latest progress is about `3.03kit/22.1kit`, recent rate recovered to about `6.8s/it`, remaining about `36h04m`, recent loss around `2.70`. A previous `29.6s/it` ETA spike was transient after eval/checkpoint work and did not persist.

Focused recent-log scan found zero real OOM, out-of-memory, HBM allocation, `RESOURCE_EXHAUSTED`, traceback, runtime exception, dead-node, owner-death, killed-process, nonzero-exit, failed-precondition, or no-accelerator signatures. Harmless checkpoint lines of the form `Error check finished successfully` were not treated as fatal signals. No patch, retry, or CC escalation is needed from this poll.

## 2026-06-24 15:05 PT Issue 6607 Public Update And OLMoBaseEval SC Retry

Updated the coordinating GitHub issue layer for #6607 to make the previous research-planning stub actionable and self-contained. The issue now points to:

- Fieldbook experiment: `exp_01kvvvv6zxrf0j7tkp4f7k6y66`.
- Launcher: `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py`.
- Reference outputs: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_baseline_mixtures_issue6607_20260623/`.
- Training parent: `/calvinxu/dm-delphi-baseline-mixtures-issue6607-remaining-20260623-2337`.
- GCS analysis artifact: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355`.
- GCS output prefix: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/`.

Resolved W&B run links for the eight baseline training datapoints:

| mixture | target FLOPs | W&B |
| --- | ---: | --- |
| proportional | `3e18` | https://wandb.ai/marin-community/marin/runs/proportional_3e18-ebc4aa |
| UniMax-8 | `3e18` | https://wandb.ai/marin-community/marin/runs/unimax8_3e18-cb3b49 |
| proportional | `2e19` | https://wandb.ai/marin-community/marin/runs/proportional_2e19-c5dbac |
| UniMax-8 | `2e19` | https://wandb.ai/marin-community/marin/runs/unimax8_2e19-b4704b |
| proportional | `3e20` | https://wandb.ai/marin-community/marin/runs/proportional_3e20-e5fca6 |
| UniMax-8 | `3e20` | https://wandb.ai/marin-community/marin/runs/unimax8_3e20-c21bce |
| proportional | `1e21` | https://wandb.ai/marin-community/marin/runs/proportional_1e21-2f1a48 |
| UniMax-8 | `1e21` | https://wandb.ai/marin-community/marin/runs/unimax8_1e21-d685cd |

Iris remains the execution source of truth for preemptible restarts. The W&B run state for restarted children can lag or show a crashed prior attempt while Iris has already launched a replacement child under the same deterministic run identity.

Also fixed the current OLMoBaseEval Easy SC gap. The full skip-existing SC array `15976926` completed `714/716` rows and failed only rows `625` (`p60_del_26`) and `626` (`p60_del_27`) on Hugging Face API 429 during checkpoint `snapshot_download`. Submitted a targeted serialized retry:

- Fieldbook experiment: `exp_01kvhcwbw4jw14ff2p3jhb46vr`.
- Fieldbook job: `job_01kvxtc2dqxr6ynpwm01e1xkz1`.
- Slurm job: `15987400`.
- Array: `625-626%1`.
- Command: `ssh sc sbatch --parsable --array=625-626%1 --job-name=olmo-easy-full-retry625626 /juice4/scr4/pinlinxu/olmo_eval_canary/jobs/olmo_base_eval_full_skip_existing_completion_20260623.sbatch`.
- Current state at submission check: queued on Slurm priority with no row logs and no startup errors yet.

Fieldbook validation `val_01kvxtm53twr2tc9n3v88xa9ed` records the OLMoBaseEval retry submission. Next action: monitor Slurm `15987400`; if it succeeds, mark `job_01kvxtc2dqxr6ynpwm01e1xkz1` succeeded, clear the blocking full-array failure by lineage, and rerun the OLMo output/writeback count.

## 2026-06-24 15:07 PT OLMoBaseEval Retry Health

Slurm retry `15987400` started cleanly:

- Row `625` is `RUNNING` on `iliad1`.
- Row `626` is `PENDING` only on `JobArrayTaskLimit`, as intended by `625-626%1`.
- Row `625` passed checkpoint resolution/model initialization, reached `Provider ready`, and began OLMo-Eval scoring batches for `p60_del_26`.
- The previous failure mode, Hugging Face API 429 during checkpoint `snapshot_download`, did not recur on startup.

Fieldbook validation `val_01kvxtr6zxsg96fddd3h76c07q` records this running-health check. Continue monitoring until both rows have `metrics.json` and W&B writeback manifests, then mark the retry succeeded and rerun the `716/716` completion accounting.

## 2026-06-24 15:12 PT Issue Comment W&B Link Correction

Edited GitHub issue comment https://github.com/marin-community/marin/issues/6607#issuecomment-4794063095 to enumerate all eight W&B datapoint links directly in the comment, not only in the issue body:

- `proportional_3e18-ebc4aa`
- `unimax8_3e18-cb3b49`
- `proportional_2e19-c5dbac`
- `unimax8_2e19-b4704b`
- `proportional_3e20-e5fca6`
- `unimax8_3e20-c21bce`
- `proportional_1e21-2f1a48`
- `unimax8_1e21-d685cd`
