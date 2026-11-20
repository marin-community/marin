# Long-loop prototype

Snowball SFT → asynchronous RLOO-N (GS8) → Micro210 → public HF checkpoint and
provenance archive. The launcher takes one content-pinned run spec. Dataset
manifests and stage configurations are explicit inputs, resolved relative to it.

```bash
uv run python -m experiments.post_training.long_loop \
  --spec experiments/post_training/long_loop/specs/round1.yaml \
  --validation-output /tmp/long-loop/full --version 2026.11.10
```

Add `--run --max-concurrent 1` for the real planned-budget run. No extra GPU
smoke, probe or qualification job is scheduled. The resolved JSON records the
spec digest, complete configs, dataset identities, scale changes and blockers.
Strict SkyRL composition runs on the actual graph documents before allocation.
Evaluation input preparation runs on CPUs and records the pinned task and rollout
specification hashes. Environment startup and agent installation happen in the
actual evaluation trials; preparation does not run setup trials or allocate GPUs.
The complete example is intentionally **not launchable yet**: the selected Smartie
cohorts still need their pinned exclusion evidence. The current SWE-only runtime
bundle supplies it; CalibForge is excluded from that run.
Use `--through sft` for the admitted SFT prefix while those checks finish.
Before submission, copy the spec inputs into a relative workspace directory and
pin `marin_revision` to the clean code commit; Iris bundles those explicit inputs.
Use a relative or cluster-local validation output path for a remote run.
A printed blocked plan is not a successful dry-run or permission to launch.

## Exp3 smoke and recipe capture

Production entrypoints refuse both tracked and untracked source changes. Commit
the complete runtime before resolving or submitting a production spec. The
archive automatically writes `recipe-bundle/recipe.tar.gz`, its SHA-pinned
manifest, and a verification receipt. It includes the canonical Iris workspace
files, consumed configurations and dataset referents. Source bytes must still
match the snapshot captured while constructing the graph. The archive verifies
every bundle member before sealing and retains the bundle in its object inventory.

The explicitly approved Exp3 wiring smoke uses 8 H100, three updates of a tiny
Snowball model, and complete short documents from the frozen Exp1 caches. It
checks CPU/GPU initialization parity, retained FP32 parameters, update norms,
chat/full-token masks, lookahead padding and native telemetry. It does not measure
the 67B initializer's upstream loss or qualify production throughput.

```bash
uv run python -m experiments.post_training.long_loop.exp3_smokes \
  --source-revision "$(git rev-parse HEAD)" --h100-hour-limit 4 \
  --version YYYY.MM.DD.N
```

Inspect the printed graph before adding `--run --max-concurrent 1`. The stage
guard charges timestamped build and worker allocation, excludes queue time,
retries watchdog stalls at most twice with fresh child names, and cancels only
the active stage when its required H100-hour ceiling is reached. Other failures
are reported without replay. Its ledger records observed spend and the ceiling.

The separately approved initializer proof runs two short updates of the pinned
67B model on two eight-H100 nodes (EP8/CP2), after a 262,144-context C4-en check.
Its provisional reference is an explicitly labeled estimate, 2.313741889 nats
per token with absolute tolerance 0.5. The frozen input audit must match before
the loss comparison. Raw and token-reseeded initialization are checked separately.
The proof keeps capacity factor 1.0 and reports route drops, as in the baseline.
Optimizer state is offloaded between steps, as in Exp1. Only the final native
checkpoint is saved; time-policy saves are disabled. A complete hash-pinned
regional C4 bundle is required; workers do not read the original GCS cache.

```bash
uv run python -m experiments.post_training.long_loop.exp3_initializer_smoke \
  --source-revision "$(git rev-parse HEAD)" --h100-hour-limit 12 \
  --reference-bundle s3://REGIONAL-PREFIX/manifest.json \
  --reference-bundle-sha256 MANIFEST_SHA256 \
  --version YYYY.MM.DD.N
```

The `snowball_exp3` SFT profile requires a hash-pinned supervision sidecar,
explicit trainer/model/mesh configuration, initial-evaluation gate and stage
guard. Scheduled SFT validation data is explicit and separate from the upstream
initialization gate's heldout data. It uses retained FP32 parameters, BF16 computation, MuonH5e-5 with
AdamLR3/13 of that, clipping1, z-loss1e-4 and the unchanged4750-step schedule.
Packing binds the sidecar's precomputed prediction weights for every source.
The full candidate catalog and recipe require separate approval before launch.

Historical recipe backfills use the CPU-only `backfill_recipe` entrypoint with
an exact archive-manifest hash, full source commit SHAs and a fresh output path.
It reads only recipe categories, verifies their archived bytes, and records
reconstruction limits. It does not execute producers or modify the sealed archive.

## Shared SFT and historical reproduction

The historical `snowball_h100` adapter uses `experiments.sft.launcher.SFTSpec` and the checked-in
`experiments/sft/configs/snowball_h100.py` defaults: native Levanter Snowball,
Adam, 32,768 positions, batch eight and one transcript per sequence. The run
spec supplies the prepared data, training budget, execution resources and
tracker project. Existing cache addresses remain pinned; the prepared-data
interface preserves concatenated sources and time-varying mixture weights.
Schedule boundaries are converted by token positions when the batch geometry
changes. No source is replaced with the example recipe's dataset.

The serialized-chat cache view masks system/user/tool text and message headers,
retains every assistant body in the window, and leaves pretraining replay under
full causal loss. Its one-transcript view explicitly keeps the leftmost 32K
positions of longer traces. This differs from the historical 262K packed,
full-text objective. The original caches are read without modification.

Historical `GrugRunConfig` files remain supported through a separate typed
backend adapter in the same shared launcher. A fresh legacy SFT launch requires
`--i-know-i-am-using-a-deprecated-config-format`; missing acknowledgement fails
during CPU configuration validation. The adapter preserves the original data,
optimizer and model configuration, binding only the new run's execution and
output locations. It does not convert a Grug configuration into chat defaults.
Adopting an already completed checkpoint does not rerun legacy SFT.

Native SFT records its submitted/consumed configuration and final selection,
then the shared trainer's HF export supplies the RL handoff. Handoff preparation
copies those selected weights within the same filesystem and declares the
original model's serving context bound. The Grug and native checkpoint formats
remain distinct in the archive. GPU smoke runs are explicit user-authorized
invocations with the same production spec and only its duration overrides.

The current execution policy removes all natural-long qualification, including
minimum-length observations and export blockers. Run normal batches and let actual
long inputs fail naturally. CPU config validation and data-correctness assertions
remain. The launcher checks adopted metadata on its CPU coordinator and does not
allocate a separate endpoint or sandbox probe. Historical smoke scale overrides
remain available for reproducing past records; they are not part of the current run.
The next RL geometry is TP1/PP2/CP5/EP8 with a2a and sample packing, using 80 learner
and 176 rollout H100. The original plumbing smoke tested handoffs at a smaller
context and does not establish throughput for this geometry.

The `.11` full SFT attempt inherited the old smoke's fixed initial mixture weights.
Its source components matched the reproduction, but it omitted the 250-step weight
schedule. It failed after its first update in the checkpoint sampling helper and
has not been restarted. The supplied SFT manifest restores all 19 schedule phases.
50B denotes scheduled packed positions, not 50B distinct supervised tokens.

New jobs run on RNO2A at **production** priority, with at most **256 H100**.
Missing inputs and data/config correctness failures stop the run. A failed stage is not
replayed automatically. Keep this PR draft; no merge approval is implied.

## Historical plumbing evidence

The completed plumbing run verified 20 supported benchmark paths, including the
BixBench repair. This was a runtime test, not a full Micro210 performance result.

The historical Micro210 runtime originated from #10933. Exact source and dataset
pins, consumed configs and retained traces are in the verified archive evidence.
The selected plumbing model is public at
[Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1](https://huggingface.co/open-athena/Snowball-LongLoop-Plumbing-Step3-2026.11.09-r1),
revision `638e48507118e60d09b64be50230f795a50deff9`.

Campaign status and evidence live in
`/Users/benfeuer/Documents/experiments/active/long-loop-prototype/`:
`STATUS.md`, `DESIGN.md`, `artifacts/archive-hf-smoke-20261109/`, and
`artifacts/run-spec-20261109/`. The original copy-based archive job was cancelled;
the replacement HF publication and evidence archive verified. Historical trace
sampling and telemetry gaps remain documented there.

## RL-only continuation

After SFT commits, pin its selection, checkpoint metadata/manifest, consumed config,
source lineage, and payload inventory in an `adopted_sft` file. Reference that file
by path and SHA-256 in the run spec. Use a fresh runtime version. Missing or changed
checkpoint evidence fails the adoption stage; it never starts a replacement SFT.

To reuse a completed HF handoff, also supply an `adopted_handoff` file pin with
its `uri`, `source_checkpoint` and `files` (URI/SHA-256 pairs for the committed
export manifest, request, config, tokenizer and index). The graph verifies these
bindings and adopts that export instead of running conversion again. CP training
requires sample packing. Preflight runs both strict composition and SkyRL's
runtime configuration validator before allocation.

```bash
uv run python -m experiments.post_training.long_loop \
  --spec /absolute/path/to/continuation-spec/round1.yaml \
  --through archive --version 2026.11.10.10 \
  --validation-output /absolute/path/to/continuation-validation
```

This constructs the full planned RL, evaluation and archive graph. Add `--run` only
when replacing the previous allocation has been reconciled. It retains context,
GS8 and exact 32/32 consumed-group quotas without a stochastic-length qualification
or an extra GPU probe. The aggregate allocation remains at most 256 H100.
CPU tests cover quota-buffer behavior, 102,400-token collation and checkpoint
adoption. Actual learner execution provides the GPU behavior evidence.

## Evaluation and archive evidence

The loop sets `run.finalization: any_terminal`: once the RL child is terminal,
the launcher exports the trainer-format checkpoint selected by validation-reward
EMA after a completed save. This applies if RL failed, was cancelled, or was
terminated after a stall. The evaluation, report and archive graph follows a
successful export for every terminal outcome. Terminal status and failure details remain unchanged in the training
record, archive provenance and timing report. If no native best checkpoint was
committed, `finalization.json` records the skip, and downstream stages write
`finalization-skipped.json` before dispatching any evaluation or archive work.
Export failures still propagate; they are not treated as a missing checkpoint.
This continuation requires the parent coordinator to remain alive; cancelling
the coordinator itself cannot run its remaining graph.

Evaluation preserves the pinned suite's published repetitions: 200 supported tasks
produce 440 attempts, and the one-task-per-path smoke produces 44 attempts across
20 paths. `selection.json` records task IDs, repetition indices, runtime row IDs
and source names. Completion requires every selected attempt, not just an equal
row count. The selected model's actual config passes the pinned vLLM context
resolver before judge allocation.

The archive derives its inventory from committed stage outputs and consumed
configs, including retention ledgers, exact evaluation selection and native result
rows. SkyRL writes the initialized W&B identity to `exports/wandb-run.json`.
New native long-loop SFT and RL runs require explicit W&B entity and project values.
Every SFT rank and the RL metric producer append each metric call to a local
JSONL file and fsync it before forwarding it to W&B. Repeated step IDs and calls
with W&B `commit=False` remain separate
JSONL records. Files live beneath `$IRIS_OUTPUT_DIR` when it is available and
are published to each stage's durable `telemetry/` prefix on normal exit or a
handled training failure.
A start marker registers each attempt. The immutable completion receipt records
SHA-256 hashes of metric-history and phase-journal files, metric row counts, and
observed step IDs. Failed attempts
remain separate. A registered attempt without a completion receipt prevents
telemetry certification.

Native journals record model initialization, data loading, compilation, first
and steady steps, checkpoint enqueue/write/drain, and HF export. RL's inference
engine initialization span includes startup and compilation; it does not isolate
compiler CPU time. The selected RL checkpoint exporter writes its own export
journal. The archive requires all expected SFT ranks, the RL metric producer and
the selected exporter, checks every payload hash and the metric record sequence numbers, and requires
matching start and successful-end events for each required phase. It copies these artifacts without querying
W&B history. Missing or changed native evidence fails archive construction.

Historical adopted stages without producer telemetry still use pinned
`additional_evidence` and the W&B history-export path described below. Use `kind: wandb-run-reference`
for an independently verified SFT identity and `native-sft-journal` /
`native-sft-attempt-archive` for preserved timing. Incomplete historical evidence
keeps the archive partial and blocks checkpoint cleanup. Model exports go to
public Hugging Face repositories; native optimizer payloads retain location and
TTL references.

Final evaluation preserves each task's declared verifier timeout. A global
300-second override previously truncated the 1,800-second
`mcmc-sampling-stan` and 7,200-second `sam-cell-seg` verifier budgets. The outer
trial deadline is 10,800 seconds to include setup, the 1,800-second agent window,
verification and teardown. Verifier exceptions receive no score and are excluded from reward means.
Whole-agent replay and score imputation are disabled.

An explicitly approved partial evaluation can pass an exact list of
`data_source`, `uid` and `exception_type` records to
`report_step(accepted_unscored_attempts=...)`. Every selected identity and each
source's expected attempt count must still match. The report retains `all_verified=false`, excludes
unscored attempts from reward means, and records the accepted exceptions. The
archive preserves this limitation, publishes an `infra_failed` EvalDash record
with scored coverage, and blocks deletion of superseded native checkpoints. Unexpected failures still
stop finalization.

For a completed evaluation with a verified supplemental BixBench result, use
the tail-only entrypoint. Its SHA-pinned JSON plan names the original resolved
run specification, adopted artifact roots, verifier-recovery manifest, exact
accepted unscored identities and additional evidence. Pin the postprocessing
commit separately from the original training revision.

```bash
uv run python -m experiments.post_training.long_loop.postprocess \
  --plan-uri s3://<bucket>/<owner>/postprocessing/plan.json \
  --plan-sha256 <sha256> --version <fresh-version>
```

Inspect the dry plan before adding `--run`. This entrypoint adopts all producing
stages and runs only the recovered view, report, SFT accounting and CPU archive.
It archives the original native rows and aggregate alongside the scored view.
The original CPU resources, deadlines and checkpoint-retention controls apply.
To recover only accounting/archive, include the committed `recovered_eval` and
`report` roots in the plan's adopted artifacts; those transformations are then
reused. Accounting resolves concatenation offsets and remaps mixture indices
in batches without changing sample order, token hashes or supervision masks.
An `accounting_resume` pin can reuse a contiguous initial JSONL sample range:
declare its `source`, `sha256`, `rows`, `consumed_config_sha256` and
`checkpoint_metadata_sha256`. The checkpoint and config must match, and only
the remaining global indices are reconstructed. The report records the prefix.

If the strict W&B exporter reports `IncompleteRunHistoryError`, the archive
confirms that typed provider error and automatically streams every row from
unfiltered, unsampled `run.scan_history()`. The JSONL and its SHA256 receipt are
archived with observed row count and `_step` bounds in the manifest. This fallback
does not certify provider completeness or permit checkpoint deletion. Other
export/provider failures propagate. A running coordinator uses an immutable
source bundle; older runs need archive-only postprocessing to use a later fix.
The archive recipe pins `wandb.package`, `wandb.sdk_version` and `wandb.base_url`.
Both export paths use that API origin explicitly: newer SDK defaults can target
a different service from the one hosting the recorded training runs.
The isolated exporter starts its own `wandb-core` process; it does not inherit
the parent's `WANDB_SERVICE` connection across SDK versions.

### CPU data and configuration validation

Strict composition checks literal source-route keys survive translation. Before
submitting the real RL run, also run the actual CPU dataset and rollout-context
setup on its generated `compiled.yaml`:

```bash
vendor/skyrl/.venv/bin/python -m experiments.post_training.long_loop.cpu_runtime_probe \
  --compiled /path/to/compiled.yaml --data-root /path/to/pinned/parquets \
  --output /path/to/cpu-runtime.json
```

Stage the exact export tokenizer at the compiled model's local tokenizer path;
model weights are not needed. Supply the ordinary configured secrets without
printing them. The probe loads and lowers all training rows, applies the runtime
token-length filter, constructs `TrainingContext`, collates every source type,
and exercises the real quota planner with scripted GS8 rewards. Its receipt
binds compiled config and local data hashes. Check those data hashes against the
run's pinned manifests before submission. This proves CPU setup and quota
accounting, not GPU learning or live grading.


The real RL job creates its own sandboxes. No separate sandbox or serving probe
is allocated before it.


The next-run recipe enables truncated importance sampling with the `tis` preset
(token ratio cap2.0), policy Router Replay and serving-side expert-ID capture.
The exact-token-ID path remains required for both raw completions and installed
agent chat. Learner router fusion is disabled so replay hooks remain active.
These settings do not alter the already submitted `.10.10` run.


The next RL harness is Pi0.87.0, matching Micro210's installed-agent runtime.
Prepare its SWE packages with `python -m experiments.post_training.long_loop.pi_rl_data`
using an input directory, its expected manifest SHA and a fresh output directory.
The transformation reuses the frozen Pi layer, selects the non-root
`longloop-agent` account and keeps task commands on the task's original Python.
Its package directories permit group installs; its scripts directory uses the
sticky bit so new console entrypoints can be installed without deleting existing
root-owned control executables. Pi's Node packages remain isolated under `/opt`.
Agent-only HOME selects Pi's writable configuration; PATH and VIRTUAL_ENV retain
the task's values, as do the verifier's interpreter and environment. `/testbed` stays
editable; its parent is root-owned. Task identity, instructions, private tests,
source revisions and row order are preserved and bound by per-row hashes.
Only SWE rows receive this overlay. Bind the derived manifest in the next run
spec; the running `.10.10` and the Micro210 task permissions remain unchanged.

The v2 private agent virtualenv isolated installed task dependencies from the
verifier. The v3 overlay removes that interpreter split. Its CPU container guard
invokes Pi's actual bash tool to install an offline wheel, then imports the module
and runs its console entrypoint through the unchanged root verifier.

Model-owned turn, token and agent-time limits retain the task verifier's actual
reward and training eligibility. Pi's generic nonzero exit needs the engine's
matching model-limit stop reason; a network or cleanup error does not qualify.
Missing or failed verification remains masked. The turn limit stays at128 and
best-checkpoint selection requires at least 90% of the expected validation
tasks scored (108 of 120 for this shape). An ineligible validation warns
loudly, is recorded with its reason in the selection receipts, and is skipped;
it never silently cancels best selection and never aborts training. When no
validation is eligible, export falls back to the final checkpoint with a
durable warning; when an eligible validation exists, the best-by-validation
checkpoint is required and its absence fails closed.


Stage deadlines follow the controlled envelope described below. RL has no
wall-time callback unless its pinned stage document explicitly supplies:

```yaml
wall_clock_budget:
  seconds: 36000
  drain_margin: 1800
```

That optional clock starts at the first optimizer completion. It stops at a
completed-step boundary after the budget expires; it does not subtract startup,
a drain reserve or a minimum number of steps from training. An in-flight step
may finish before finalization. The launcher rejects a task/coordinator timeout
that does not exceed budget plus drain margin. Overrides remain explicit in
`rl.execution.job_timeout_seconds`, `evaluation.execution.job_timeout_seconds`,
`execution.coordinator`, or `execution.task_timeouts` for SFT/export/eval-inputs/archive.

Future RL validation runs every two optimizer steps. Checkpoint selection uses
five complete validation observations at that cadence, with missing observations
rejected; the same best checkpoint goes to Micro210, HF and archive.
Historical `.11.2` uses its immutable eight-hour callback, nine-hour task deadline
and every-step validation. These source defaults do not modify that job.

### Explicit stage settings

The pinned [RL](specs/rl.yaml) and [evaluation](specs/evaluation.yaml) documents
supply context and output limits, Pi configuration, decoding parameters, geometry,
batch sizes, optimizer settings, validation cadence and task limits. The launcher
validates [required fields](stage_config.py) before SkyRL composition and binds
model/data artifacts and output paths. It does not supply experiment
hyperparameters. The [CPU regression](tests/test_run_spec.py) builds the actual
artifact graph, emits its launch YAML, runs the production SkyRL composition, and
compares every supplied setting with the compiled configuration, excluding artifact
bindings such as model and data paths.

The RL document explicitly selects `daytona_image_build: snapshot` for training
and `eval_daytona_image_build: declarative` for inline validation. Declarative
validation creates sandboxes directly from each task's image or build context
with the eval credential, without creating or leasing named snapshots. Final
evaluation also selects declarative provisioning in its evaluation document.

The [next-run spec](specs/round1.yaml) explicitly uses temperature 0.7 for training,
held-out validation and final evaluation, and 16,384 output tokens per turn. In the
stopped `/benfeuer/long-loop-full-2026.11.11.2`, training temperature is 1.0,
validation temperature is 0.0, and the RL per-turn output limit is 8,192; final
Micro210 has its separate 16,384 limit. The next-run settings are user-selected;
no measured improvement from the new temperature is claimed. That trial was stopped after the conversation-boundary audit.

Episode metrics distinguish all generated tokens from optimized response tokens.
A masked trial cannot count as a pass; it remains in the attempted-task denominator.
Scored coverage is the fraction of attempted trials with an unmasked verifier
score. `raw_verifier_pass_at_N` retains raw verdicts for diagnosis.

The `.11.2` trace audit found that Pi reserialized assistant messages and tool-call
JSON. The strict history comparison treated those changes as a new conversation
on each turn, leaving one turn in the selected final conversation. The repaired
[Harbor adapter](../../../vendor/skyrl/skyrl-train/skyrl_train/rollouts/harbor_trial_session.py)
compares message semantics, including parsed tool arguments and empty optional
reasoning fields. Exact-token continuation keeps the tokens originally served;
it does not reconstruct earlier actions from Pi's reserialized text. A
[CPU regression](../../../vendor/skyrl/skyrl-train/tests/cpu/test_pi_conversation_continuity.py)
uses captured Pi 0.87 round-trip messages for 60 turns, verifies one conversation
and loss on every assistant action, and verifies a genuine compaction starts a
new conversation.

The next trial explicitly budgets 10 hours from its first optimizer step. Native
checkpoints save every two updates and retain the latest three plus the selected
best; the selection uses five validation observations at two-update intervals.
Final evaluation, HF export and archive use that same selected best checkpoint.
Archive cleanup reads `trainer.max_ckpts_to_keep` from the consumed RL launch and
preserves that rolling window plus the selected best. Its sealed inventory names
the retained checkpoints, and cleanup rejects any retained path.

Post-run archival and pruning never remove the final SFT checkpoint. The archive
plan records that checkpoint from the SFT stage's `selected.json`, cleanup rejects
any candidate at or under it, and after cleanup the archive verifies the
checkpoint again: permanent metadata, weights plus optimizer state in the
manifest, a complete OCDBT chunk inventory, and sample decodes. The verification
receipt lands in `final-sft-checkpoint.json` beside the sealed archive. The same
rule protects the RL side: the best-by-validation checkpoint and the committed
final checkpoint stay in the retention set even when newer partial directories
exist, and the post-archive receipt `best-rl-checkpoint.json` re-checks both —
committed `trainer_state.pt`, a weights payload, and a permanent path.

For a BixBench verifier timeout, a separately verified grade of the retained
original answer can be incorporated with `bixbench.merge_verifier_recovery`.
Pass the original evaluation, frozen selection, recovery receipt root and pinned
recovery-manifest SHA to a new artifact version. The merge requires the complete
selected native result set and that one recoverable unscored attempt. Pass an
explicit `accepted_unscored_attempts` list (empty for full scoring) for any
separately approved infrastructure exclusions. Unexpected missing rewards still
block the merge; accepted exclusions remain unscored in the view and report.
It checks the
original task/session, sampled trajectory, final answer, frozen grader and
verifier deadline; it does not generate another answer. The scored view preserves
every original native file and aggregate under `original-native/`, records native
versus recovered coverage, and computes view metrics in `evaluation.json`.
Its launch and terminal files are verbatim snapshots of the original job.
# SWE task admission

The repaired SWE release is derived from the pinned source packages rather than
substituting a shared environment. Its five image contexts add a harness bootstrap
and an isolated grading runtime. Bootstrap checks out the exact bug commit, runs
the upstream dependency setup, then freezes trusted grading dependencies before
the agent starts. Grading first checks HEAD ancestry, reconstructs a fresh source
worktree, restores tests and pytest configuration, and runs the task tests with
plugin autoload disabled. Agent package installations do not enter that interpreter.
Root control commands use immutable bindings captured before agent access;
grading console scripts use the frozen task interpreter rather than agent packages.
Each grade preserves HEAD, candidate patch, import origins, package inventory and
the verifier report. Candidate failures score zero; infrastructure failures remain
unscored.

`swe_admission_cluster` runs the same CPU Docker controls used locally, sharded by
repository and image context. Submit its coordinator on RNO2A at production priority
with the privileged container profile, an inherited HF secret and a three-day task
timeout. It takes the pinned input prefix and manifest SHA, output prefix, job
prefix, tasks per shard, setup timeout, verifier timeout and enclosing Harbor timeout
as arguments. All shards are submitted directly; each builds its original task
image and runs the controls with private userspace networking. This uses no RL
allocation or model requests.

`admission_index` commits exactly one complete evidence inventory per original
task across original and recovery shard plans. It rejects missing or duplicate
commits and references existing evidence without copying it. Interrupted,
uncommitted task outputs remain preserved but cannot certify admission.

`seal_swe_admission` requires this complete index and exactly one receipt per
original SWE task. Only a scored unfixed-bug zero and scored reference one admit a
task. Failed, degenerate and unexecuted controls cannot qualify it. Sealing records
before/after identities, observed dependency inventories and exclusions, preserves the nonagentic
validation rows, and explicitly records any seeded replacement of rejected SWE
validation rows. Its output is a new immutable dataset version; existing releases
and stopped-run evidence remain unchanged.
The published task package is exactly the one whose controls ran. Sealing adds
no untested dependency constraints; the observed package versions remain available
for reproduction, while each new trial freezes its own grading environment.

## Controlled stage allocations and deadlines

New loops use 256 H100s for SFT, 80 learner plus 176 rollout H100s for RL,
and 128 candidate plus 8 judge H100s for final evaluation. Input preparation,
HF handoff export, accounting and archive use CPUs. These controls are defined in
[execution_policy.py](execution_policy.py) and checked before submission.

The stage deadlines excluding final evaluation sum to 36 hours: inputs 30 minutes,
SFT 20 hours, handoff export 1 hour, RL 12 hours, evaluation preparation 30 minutes,
SFT accounting 30 minutes and archive 90 minutes. Final evaluation has a separate
72-hour deadline. The enclosing coordinator has 109 hours so it cannot undercut
the serial deadlines. The nominal 36-hour experiment target still includes
evaluation; the longer eval deadline allows slow evaluations to finish.

Changing a stage allocation or deadline requires both
`--i-know-i-am-breaking-experimental-controls` and
`--experimental-controls-reason "reason for the deviation"`. The resolved spec
and archive provenance record the baseline, effective values and decision.
A smoke shortens updates while retaining the SFT topology, context, packing and
optimizer. It must complete a native checkpoint and HF export before its timing
is used to size the full run.

Experiment 2 explicitly selects packed 262,144-token context, batch 8 and a
CP32/EP8 mesh on 256 H100s. Adam and the pinned historical mixture remain unchanged.
The archive includes the population truncation/padding census and an exact-loader
replay of the committed SFT sample, with per-source counts, token hashes, loss
coverage and estimates of packing efficiency.

Experiment 2 sets `sft.xla_flags: [--xla_gpu_shard_autotuning=false]` for
training and its native validation callbacks. Smoke runs completed updates but
blocked in XLA ConfigAssigner's cross-host autotune-result exchange during
validation. Each host now autotunes independently; compilation, autotuning and
CUDA graphs stay enabled. Cold tuning work may increase. The requested flags
and accelerator environment are recorded in `consumed-config.json`; the run
spec and that receipt are archived. This setting does not change the RL or
final policy-evaluation runtime.

For an RL-only recovery after SFTSpec finishes, pin `adopted_sft` and
`adopted_handoff` to the committed SFT selection and exported checkpoint. The
adoption verifier checks the native zero-based final step against completed
updates, the exact consumed config, model/tokenizer lineage and payload inventory.
It uses SFTSpec's consumed model defaults rather than the legacy Grug recipe.

If RunAI object streaming fails during engine startup, set
`rl.config.generator.model_loading: stage_local` in a freshly pinned run spec.
The existing staging path materializes the same checkpoint once per node before
Ray startup and vLLM loads local safetensors. Reserve enough node-local disk for
the full checkpoint; the policy's source URI and identity remain in the consumed
config. This changes weight loading only; compilation and CUDA graphs stay enabled.

## Exp3 production preparation

`exp3_production` consumes one final data-release URI and overlays its pinned
catalog, supervision metadata and natural weights on the frozen Exp1 RL recipe.
It fixes SFT at 256 H100 (EP8/CP32), batch 256, context 262144, maximum 746 updates and
a 20-hour execution deadline. The trainer reserves time for the final native/HF
checkpoint and records the actual completed count. Source must be clean and
pinned to the submitted commit.

Future Exp3 recipes use the mainline first-fit packer from #11379 through
`pack_lookahead_window: 128` with `pack: 256`. This changes the recipe's pack
membership from the former largest-fitting-document policy. Packed prebuilt
caches now left-slice over-long documents as singleton packs; the former packer
rejected them. Rebuild accounting and use a new artifact version for this recipe;
already submitted runs retain their pinned packer and recipe.

Future production recipes save rolling native SFT checkpoints every two hours,
retaining three resume checkpoints. Finalization forces a permanent checkpoint
even when the last periodic save was recent. The trainer drains the write and
reads its committed metadata before publishing the training completion receipt.
SFT selection verifies that receipt and checkpoint again; the HF handoff and RL
consume the selected final update. Graceful time-limit and maximum-update stops
use this path. Exhausting the dataset also preserves the last completed update,
but fails controlled-run admission. This policy does not alter already submitted
jobs: Exp3 artifact `2026.11.16.51` retains its pinned 30-minute interval.
Retention keeps the final SFT checkpoint at its permanent path for the run's
whole lifetime; rolling temporary checkpoints may still be pruned.

Before requesting approval, run the graph preflight with no `--run`:

```bash
uv run python -m experiments.post_training.long_loop.exp3_production \
  --data-release "$DATA_RELEASE_URI" \
  --source-revision "$RUNTIME_COMMIT" \
  --version "$ARTIFACT_VERSION" \
  --validation-output /tmp/exp3-production-preflight \
  --max-concurrent 1
```

A CPU-only release (`training_ready:false`) can validate configuration wiring;
it is rejected if submission is requested. Final readiness requires the release
publisher's catalog, weights, retention tables, native consumer receipt and
Atlas evidence. The release never substitutes for user approval of the recipe,
catalog and retention table. All referenced metadata bytes are checked against
SHA256 pins; cached token bodies and model weights are not downloaded by this
local preflight. The resolved plan and concrete graph validation receipts are
written to `--validation-output`.

`--through sft` limits preflight to the SFT stage when a downstream control
remains unresolved. Full-loop admission still enforces the RL learning-clock,
drain and outer-job deadline relationship. A successful SFT-only dry run does
not certify downstream admission. After approval, the same frozen command with
`--run` submits through Iris; keep the runtime commit and artifact version
unchanged between approval and submission.

## Defining a new round

Round configuration lives in YAML under `experiments/post_training/long_loop/rounds/`;
code holds the mechanism and the invariants. `round_runner` loads a round YAML
into a strict schema (unknown or mistyped fields fail), resolves its named
references through the code registry (data release, curriculum pools, evaluation
policy, initializer), re-derives the resolved plan through the shared pipeline,
and enforces the shared invariants: whole-run annealing (optimizer horizon equals
the updates actually run, final LR near min_lr), the special-token LR multiplier
default of 1.0 with explicit-override semantics (R2), a permanent output prefix,
and the final-SFT-checkpoint retention rule (R1, enforced again at archive time).

To define a new round:

1. Copy the previous round's YAML, for example `rounds/exp5.yaml` to
   `rounds/exp6.yaml`.
2. Change the identity fields: `name`, `version`, `slug`, `storage.output_prefix`
   and the `archive` prefixes. Change recipe knobs only when the round intends
   them: `sft.updates`, `sft.batch_size`, `sft.min_lr_ratio`,
   `sft.learning_rate`, `sft.adam_learning_rate`,
   `sft.special_token_lr_multiplier`, `sft.mesh`, `sft.resources`, `cluster`,
   and the registry bindings `data_release`, `pools`, `evaluation`,
   `initializer`. The runner requires `--version` to equal the YAML's version.
3. Run the dry run in region (no `--run`; the Mac cannot read the pinned
   storage inputs):

```bash
uv run iris --cluster=cw-rno2a job run \
  --job-name round6-preflight-dryrun --priority interactive --timeout 21600 \
  --cpu 8 --memory 16GB --enable-extra-resources --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.post_training.long_loop.round_runner \
    --round experiments/post_training/long_loop/rounds/exp6.yaml \
    --version "<the YAML version>" \
    --source-revision "<the runtime commit>" \
    --validation-output '$IRIS_OUTPUT_DIR/round6-preflight'
```

The dry run writes `resolved-plan.json`, `optimizer-groups.json` and
`sft-lr-schedule.json` plus per-stage launch validation receipts to the
validation output, and prints the resolved artifact plan. Submitting the round
(`--run`) is a separate, explicitly approved step; keep the runtime commit and
the YAML version unchanged between approval and submission.

## NVL72 profiles

Use this package's artifact main for adopted Snowball and Mochi RL inputs:

```bash
uv run python -m experiments.post_training.long_loop \
  --spec /path/to/nvl72-run.yaml --through rl --smoke \
  --version YYYY.MM.DD.N --validation-output /path/to/preflight
```

The command validates without submitting. Add `--run` after the checks pass.
The spec uses `format: nvl72_rl` and names `marin_revision`, `output_prefix`,
`profile`, `model_uri`, `tokenizer_revision`, `data_uri`, `data_sha256`,
`input_version` and `max_steps`. Use an immutable Git revision, calendar input
version and regional `s3://marin-us-east-08a/` output prefix. `data_uri` is a list
with one adopted artifact directory; `data_file` defaults to `train.parquet`.
`tokenizer_uri` defaults to the model input. Normal runs also require a
`validation_data_uri` list; its default file is `validation.parquet`.

Mochi profiles are `one-rack` (48 trainer + 12 inference GPUs, 65,536 context)
and `three-rack` (64 trainer + eight 16-GPU inference engines, 262,144 context).
Snowball parity profiles are `snowball-one-rack` (32 + 8 GPUs) and
`snowball-three-rack` (64 + two 64-GPU engines). Snowball uses PP2 because its
26 layers cannot use Mochi's PP12 or PP16. The profiles request physical NVLink
domains at production priority. Compilation and piecewise CUDA graphs stay on.
Both Snowball profiles select the InfiniBand ports `ibp0:1` through `ibp3:1`
explicitly and use a 4-MiB NCCL buffer on mixed InfiniBand/RoCE hosts.

NVL72 coordinators use `mochi-<profile>-<version>` for RL and
`mochi-<evaluation-name>-<version>` for policy evaluation. Iris adds the submitting
user namespace, giving the cancellation prefix `/<owner>/mochi-`. Children stay
beneath that coordinator's job ID. Record the exact submitted IDs before cancellation.

Smoke mode uses the smallest even number of groups that covers policy data
parallelism: four for Snowball and two for Mochi, with eight samples per group.
It uses two rollout workers and a one-hour clock after the first optimizer update. It saves each update and at finalization;
it skips held-out evaluation and best selection. Normal runs save every two hours,
at finalization and when best selection commits an improved checkpoint. They
retain the 90%-of-expected coverage gate (108 of 120 for this shape) and the
five-observation evaluation EMA seeded by the first eligible validation.
Native checkpoints are not pruned by these profiles (`checkpoint_count: -1`);
the regional temporary storage has a 14-day lifecycle. Record a durable handoff
before that lifecycle expires. No model weights are copied to the submitter.

For an adopted NVL72 run, upload the small self-contained run spec to regional S3
and pass its immutable URI to `--spec`. The coordinator replays this argument on
Iris; a Mac-local absolute path is not available there. Keep `--validation-output`
under `/tmp` when submitting. Save its local preflight receipts in the experiment
record before launch.

### Adopted NVL72 policy evaluation

Use the same main with `format: nvl72_eval` and `--through eval` to evaluate an
existing HF export. Omit `--smoke`; the supplied micro dataset defines the scope.
Run without `--run` first. The spec pins the export manifest, model config,
checkpoint step, tokenizer revision and evaluation Parquet SHA. A changed pin
fails before allocation. `NVL72EvaluationSpec` in `nvl72_evaluation.py` defines
all required fields.

This path uses SkyRL's generation entrypoint with no training dataset. The
policy uses one EP12 inference pool on three four-GPU GB200 hosts. The shared
judge uses TP4 on one further host. Both use production priority in 08a;
compilation and CUDA graphs stay enabled. The ordinary GB200 gang placement
binds each pool to an NVLink domain. Keep the same context, sampling, task
selection and judge pins when comparing two policies. The result has no RL
checkpoint or synthetic training-completion marker.

### Mochi Micro210 evaluation admission

Both adopted Mochi evaluation rounds declare their three IB transport variables and
4-MiB NCCL buffer explicitly. The historical 256-KiB serving qualification
predates the NVLSTree corruption finding in #11649. Compilation and PIECEWISE
CUDA graphs remain enabled. Exact CPU preflight checks each local model role's
TP × PP × DP rank count against its physical GPU allocation. API judge
credentials and live model availability are checked before rollout allocation.

The gate-2 base/SFT evaluations using the earlier frozen MiniMax policy are
preserved at `benfeuer/mochi-micro210-gate2-frozen` (commit `da0663fbea`). That
source includes the self-hosted TP8 judge receipt and terminal checks; current
rounds use the versioned API judge policy.

### Fresh Exp3 RL after the Socket failure

`exp3_rl_restart` adopts the sealed `.51` SFT export and data inputs without
rerunning SFT. Supply the hash-pinned predecessor plan and launch document as one
reference JSON object. The fresh lineage removes forced Socket transport, keeps
the 12-hour learning budget, saves native checkpoints every two hours and at
finalization, and labels a final-native HF fallback when no native best exists.
Evaluation sandbox storage is 128 GiB; training sandbox resources are unchanged.

Run without `--run` first, with `--reference`, `--reference-sha256`,
`--source-revision`, `--validation-output`, and a fresh `--version`. Inspect the
resolved launch diff and the cross-node InfiniBand probe receipt before adding
`--run`. Both the coordinator and child job names use the `mochi-` prefix.

## Consolidated adopted rounds

Use `round_runner` for the Mochi profiles in `rounds/mochi-one-rack.yaml` and
`rounds/mochi-three-rack.yaml`. Both adopt the pinned schema-3 SFT export and
regional train/held-out pools. Their graphs contain no SFT producer. Model
metadata and data-file SHA-256 pins are checked before the training allocation.

| Round | Trainer | Inference | Context |
| --- | --- | --- | --- |
| Mochi one rack | 48 GB200, TP1/PP12/CP4/EP4 | 12 GB200, DP12/EP12 | 65,536 |
| Mochi three racks | 64 GB200, TP1/PP16/CP4/EP4 | 8 engines of 16 GB200, DP16/EP16 | 262,144 |

The three-rack profile uses FlashInfer, PIECEWISE graphs and verified expert-block
weight publication. Both profiles keep compilation and graphs enabled and use
the published BF16 SkyRL wheel `marin-vllm-gpu-20261113-0ac9dcb157ee`, pinned
in the `vllm` extra in [SkyRL dependencies](../../../vendor/skyrl/pyproject.toml).
These are configuration ports. CPU validation does not establish GPU fit, serving readiness or throughput.

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/mochi-one-rack.yaml \
  --through rl --version 2026.11.19.201 \
  --source-revision "$(git rev-parse HEAD)" \
  --validation-output /tmp/mochi-round-validation
```

The command validates without submitting. Normal runs use a 10-hour learning
clock that starts after the first optimizer update and starts again after process
recovery. The separate GPU-job limit is 48 hours. `rl.max_steps` is an absolute
optimizer-step target. Set `rl.native_checkpoint` to adopt native model, optimizer
and trainer state; same-version recovery first uses that output version's own
latest checkpoint. Keep `rl.role_plan`, `rl.megatron`, the adopted model pins,
`rl.recipe.trainer.callbacks` and `rl.selection_eligibility` fixed for recovery.

`rl.selection_eligibility` is the single eligibility choice. Its current value,
`covered`, requires verifier scores for at least 108 of the 120 held-out task
attempts. Each eligible evaluation is one observation. Evaluations run every two
optimizer updates; selection can begin at update two. The adjusted EMA uses up to
the last five eligible observations during warmup and saves every improved best
immediately. `complete_window` requires all 120 scores and five consecutive
scheduled evaluations; with the two-update cadence, selection can begin at update
ten. Missing evaluations invalidate that window.
The shipped rounds use `covered`.
Completed selection history survives same-version recovery.

Long-loop RL writes native checkpoints under the permanent artifact output at
`<output_path>/checkpoints`. The protected best and final native checkpoint stay
there through archival. Attempts, traces and Ray logs keep their TTL storage
lifetime. Adopt an existing checkpoint through `rl.native_checkpoint` to start a
new output version; the storage policy does not move existing checkpoint data.

For a proposed two-update smoke, make an explicit new round with `rl.max_steps: 2`,
a fresh version and `rl.execution.job_timeout_seconds: 7200`, then add `--smoke`.
Smoke uses two groups of eight samples, two rollout workers, no evaluation or
best selection, a one-hour learning clock and a native save on every update.
This does not authorize GPU submission. Mochi smokes depend on serving readiness.

`rounds/exp4-recovery.yaml` rebuilds the downstream RL recovery graph for version `2026.11.19.82` from a
hash-pinned plan and completed SFT/handoff records. It preserves the recorded
72-hour job limit and 12-hour learning budget for historical reproduction.
Its only recipe migration removes the unused `tis_splice` flag rejected by main;
TIS remains selected through `off_policy_correction`. The already-submitted recovery version `2026.11.19.82` is unchanged. Use `--through rl` or `--through archive` for this round.

The legacy `sft-smoke.yaml` is a 300-line source config. It expands the two pinned
JSON objects in `s3://marin-us-east-08a/marin/long-loop/config/2026.11.19/` at launch
and verifies their SHA-256 hashes. It uses the original step-zero mixture weights only. The full 19-phase schedule
remains pinned by `specs/sft-data.yaml` for production SFT. Stored parameters must
be FP32. Tests that read these production pins are CPU
integration tests; ordinary offline tests do not require object-store access.
