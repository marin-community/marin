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
The complete example is intentionally **not launchable yet**: the selected Smartie
cohorts still need their pinned exclusion evidence. The current SWE-only runtime
bundle supplies it; CalibForge is excluded from that run.
Use `--through sft` for the admitted SFT prefix while those checks finish.
Before submission, copy the spec inputs into a relative workspace directory and
pin `marin_revision` to the clean code commit; Iris bundles those explicit inputs.
Use a relative or cluster-local validation output path for a remote run.
A printed blocked plan is not a successful dry-run or permission to launch.

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

Evaluation preserves the pinned suite's published repetitions: 200 supported tasks
produce 440 attempts, and the one-task-per-path smoke produces 44 attempts across
20 paths. `selection.json` records task IDs, repetition indices, runtime row IDs
and source names. Completion requires every selected attempt, not just an equal
row count. The selected model's actual config passes the pinned vLLM context
resolver before judge allocation.

The archive derives its inventory from committed stage outputs and consumed
configs, including retention ledgers, exact evaluation selection and native result
rows. SkyRL writes the initialized W&B identity to `exports/wandb-run.json`.
For adopted SFT, `additional_evidence` accepts explicit `source`, `name`, `category`
and `sha256` pins. Use `kind: wandb-run-reference` for an independently verified
SFT W&B receipt and `native-sft-journal` / `native-sft-attempt-archive` for preserved
native timing evidence. Missing native timing coverage or strict W&B history
verification produces an explicitly partial archive and prevents checkpoint
cleanup. Model exports go to public Hugging Face repositories; native optimizer
payloads remain location/TTL references.

If the strict W&B exporter reports `IncompleteRunHistoryError`, the archive
confirms that typed provider error and automatically streams every row from
unfiltered, unsampled `run.scan_history()`. The JSONL and its SHA256 receipt are
archived with observed row count and `_step` bounds in the manifest. This fallback
does not certify provider completeness or permit checkpoint deletion. Other
export/provider failures propagate. A running coordinator uses an immutable
source bundle; older runs need archive-only postprocessing to use a later fix.

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
best-checkpoint selection still requires all120validation tasks to be scored.


New submissions default to three-day task/coordinator deadlines, from
`execution_policy.DEFAULT_TASK_TIMEOUT_SECONDS`. RL has no wall-time callback
unless its pinned stage document explicitly supplies:

```yaml
wall_clock_budget:
  seconds: 43200
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

The next trial explicitly budgets 12 hours from its first optimizer step. Native
checkpoints save every two updates and retain the latest three plus the selected
best; the selection uses five validation observations at two-update intervals.
Final evaluation, HF export and archive use that same selected best checkpoint.
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
before/after identities, dependency locks and exclusions, preserves the nonagentic
validation rows, and explicitly records any seeded replacement of rejected SWE
validation rows. Its output is a new immutable dataset version; existing releases
and stopped-run evidence remain unchanged.
