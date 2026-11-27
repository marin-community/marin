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

## Full-store SFT epochs

New round YAMLs default to `stopping: {type: mixture_epochs, count: 1}`.
The epoch reads every packed sequence from every included SFT store's declared
training portion once. Explicit IID validation reservations remain validation-only
and appear separately in the capacity and exposure receipts. Corrections and
filters belong to the pinned source version. The full-store path rejects runtime
subsampling through `max_train_batches`, `experiment_budget` or `target_budget`.

The long-loop planner adapts the October 5 Hero SFT per-block quotas while
keeping that historical recipe unchanged. It assigns exact SFT quotas.
Replay fills its declared fraction of the final mixture, measured in packed
sequences. Whole-batch rounding adds replay slots; it never repeats or drops
SFT sequences. The final mixture block can be shorter
than 64,000 sequences and is still shuffled in full. Only the training batch
rounds the run length. Every included source receives at least one integer slot across the epoch,
including tiny replay sources. The preflight rejects zero allocations or
incomplete SFT coverage. Replay need
not exhaust its stores; its declared share controls how much replay is drawn.

Measure native packed capacities on in-region CPUs before constructing the round:

```bash
uv run python -m experiments.post_training.long_loop.sft_epoch_capacity \
  --data "$DATA_URI" --sha256 "$DATA_SHA256" \
  --output-prefix "$CAPACITY_PREFIX" --version 2026.11.22.1 --run
```

`DATA_URI` and its SHA-256 identify the release's `data.json`; `CAPACITY_PREFIX`
must be a permanent prefix in the same US-EAST-08A bucket as the data. The CPU job reads document
offsets and native pack indices, without token bodies. Its `SFT_CAPACITY_COMPLETE`
line gives the immutable `capacity.json` path and SHA-256 for the round:

```yaml
sft:
  stopping: {type: mixture_epochs, count: 1}
  epoch_capacity: {path: "<capacity.json URI>", sha256: "<SHA-256>"}
  data_seed: 0
```

The round derives `num_train_steps` and the optimizer horizon from that epoch.
Training stops after the epoch or at the shared 20-hour SFT limit, whichever
comes first. The 20-hour envelope includes the default reserves of 40 minutes
for finalization and 10 minutes for the next update, leaving time to save the permanent final
checkpoint and export. Omit `training_budget` to use this default; `null` is
rejected. The LR schedule remains explicit. A wall-clock cooldown scales it
using remaining training time when configured. There is no default allocation-hour
budget. A time-bound completion records its actual exposure and does not claim
the full planned epoch was consumed. The operational task timeout remains a
separate last resort and does not certify successful completion.

Every plan and SFT output carries `sft_capacity.json` and the planned
`sft_exposure.json`. After training, `sft_actual_exposure.json` records the
packed-sequence counts through the completed updates and the unique sequences
read divided by each source's declared training capacity. An epoch that stops early
keeps its checkpoint and fails before downstream RL can treat it as complete.

## Native Mochi SFT on sealed V5

`rounds/mochi-sft-exp3-v5-128.yaml` prepares the native hero SFT on Exp3's sealed
V5 release. It restores the hero's native step 393491, including optimizer state,
and preserves the step 400407 run's model, tokenizer, MuonH/Adam learning rates,
global clip, special-token reseeding, expert placement and 262144-token context.
Two 64-GPU racks use CP4/EP16 per rack. Six 32-sequence microbatches form one
logical 176-sequence update; 16 padding slots contribute no loss or routing load.
Gradients share the logical token denominator and are clipped once before one
optimizer update. Evaluation keeps the hero's Paloma corpus and dropless policy,
with batch 32 to fit the smaller mesh.

The canonical one-SFT-epoch planner binds every included V5 SFT training pack
once, with its declared replay share and existing validation reservations.
The run targets 360 updates within the epoch, with the hero's schedule fractions
scaled from its 6916 SFT updates: zero warmup and a 36-update linear tail to the
unchanged 0.1 minimum ratio. At the conservative 175.8-second update estimate,
this leaves 95 minutes for startup and overhead after the 50-minute reserves.
The shared 20-hour clock remains a safety cap. Actual source coverage and the permanent
final checkpoint are recorded separately; an early stop does not claim a full
epoch. A clock stop skips the forced final evaluation to preserve the checkpoint
reserve; periodic evaluation stays enabled. Training reads no old hero training
caches. Native resources exclude the faulty r136 NVLink domain and inherit the
CPU coordinator's PRODUCTION priority.

The preflight requires the hash-pinned seal, publisher, current native capacity
receipt and completed regional initializer-copy inventory. It reads metadata,
not weight payloads, on the operator host. It also applies the runtime batch
validator to the target's 32-wide data mesh before GPU admission:

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/mochi-sft-exp3-v5-128.yaml \
  --version 2026.11.23.127 --through sft \
  --source-revision "$SOURCE_SHA" --validation-output /tmp/mochi-sft128
```

Submission requires explicit user approval. After approval, run the same command
with `--run` from an Iris CPU coordinator on 08a at PRODUCTION priority. Native
Grug dispatches the 128 GB200 allocation; no separate GPU launcher or smoke is used.

The fresh-optimizer reference is
`/karan/mochi-sft-bankadam-fresh-r10-128-8h-u1000-step393491-20261123`.
Its [resolved record](https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Fmarin%2Fusers%2Fbenfeuer%2Fmochi-sft-preparation%2Fbankadam128-20261124%2Freference-dafc25df.json)
(SHA256 `dafc25df2728ba470e2d82496c552e701d7d449cb7bdded8b2db3b73dfc6ab9e`)
preserves batch 32 without accumulation, replica size 1, EP16/CP4 and the two-rack data axis. A `MochiWeightsInitializer` loads only
base parameters and pending router biases, starts at update zero with fresh
moments, then applies the recorded expert placement and chat-token reseeding.
The reference batch, replica/data mesh, packing and optimizer stay fixed.
`muonh_param_ndims: [2, 3]` routes stacked 4D expert banks to Adam; other matrix
groups keep MuonH. The linear tail is `round(3400 / 34000 * updates)`;
the five-update warmup and minimum LR ratio 0.1 are retained. The common 20-hour clock still applies.

Measure packing capacity for that reference before binding the one-epoch ceiling.
The regional CPU `sft_epoch_capacity` command accepts `--lookahead 0` for
sequential packing; a positive value selects bounded lookahead. Both the packing
cap and lookahead must match the round, and the capacity receipt must cover all
sealed components. For a hash-pinned `data.json` and permanent receipt root:

```bash
uv run python -m experiments.post_training.long_loop.sft_epoch_capacity \
  --data "$DATA_URI" --sha256 "$DATA_SHA" --output-prefix "$RECEIPT_ROOT" \
  --sequence-length 262144 --packing-cap 262144 --lookahead 0 --run
```

Run this CPU artifact in the data region. Bind its receipt and
`sft.packing: {cap: 262144, lookahead: null}` in the round. The weights-only
preflight verifies the existing regional initializer metadata without requiring
or reading its optimizer payload.

## Exp6

[`rounds/exp6.yaml`](rounds/exp6.yaml) binds sealed V9 and the published Exp5 RL
curriculum. SFT keeps Exp5's 256-H100 R1 mesh, precision and optimizer controls.
Its full training epoch is 1,893 updates at batch 256 and context 262,144,
with 56 reserved validation sequences excluded. It stops after that epoch or
the shared 20-hour budget, whichever comes first. Actual exposure records
whether the epoch finished.

RL uses the 32-H100 TP1/PP2/CP4/EP16 trainer and 176 rollout H100s.
The [trainer-only probe151](https://iris.oa.dev/#/job/%2Fbenfeuer%2Fmochi-exp6-muonh-memory-cp4-ep16-20261123-151)
qualified warm memory headroom and two matched updates on Exp5 SFT527.
The new model runs native memory sizing before rollouts.
MuonH uses LR `1e-5`, Adam fallback LR `2.307692307692308e-6` and momentum
offload; Adam optimizer offload is disabled. The curriculum derives its
200-prompt holdout and 13/51 route quotas. The shared RL clock is 12 hours,
followed by canonical Micro210 and archival. Best/final checkpoint retention
and the permanent final SFT checkpoint follow the shared rules. Set `SOURCE_SHA` to the committed revision used for both
preflight and submission.

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/exp6.yaml \
  --version 2026.11.24.157 --through archive \
  --source-revision "$SOURCE_SHA" --validation-output /tmp/exp6-preflight
```

Run that command without `--run` first. Submit the same pinned round with
`--run` after the preceding round's canonical evaluation is sealed.

## Exp5 SFT release

Exp5 setup reuses qualified V7, whose four science components already derive
from the sealed accepted/repaired science release. It preserves Exp4's SWE
bug-fix pool, adds the pinned Atlas IFBench training release and base HARDMath,
and assigns five percent of RL rollout groups to OpenResearcher through its
qualified thin image and hosted retrieval service.
`exp5.py --stage sft --run` submits only
SFT and its input verification as terminal stages. Downstream submission
requires the certified SFT handoff and the published, training-ready
eleven-source pools.
Run the same command without `--run` first to validate the SFT graph and record
the resolved plan. This check does not qualify the downstream RL stages.

SFT uses the parameter mesh verified in R1's run: context-8 within each node, data-32
across nodes, and size-one expert and replica axes. Expert storage partitions
experts over `(expert, context)` and hidden dimensions over `data`. This keeps
the cross-node parameter sharding used by R1. The earlier Exp5 port put the
32-node axis in `replica_dcn`, which the expert parameter specs did not name;
it replicated those parameters across nodes and made each local expert bank
32 times larger per device than R1's sharding. Mesh qualification checks parameter shard shapes as well as
token placement.

The completed production SFT uses FP32 parameters and BF16 compute. Optimizer
updates preserve each stored parameter's dtype so updates do not change the
compiled train-step input signature.
The special-token multiplier remains `5.656854249492381`. Native Snowball
recomputes each scanned layer under `jax.checkpoint` (`recompute_all`). The
corrected hero-EP optimizer grouping (#11639) is retained: expert matrices use
MuonH, RMSNorm scales use Adam, and `lm_head`/`output_proj` use AdamH. R1's
generic-file grouping instead put experts on Adam and stacked norm scales on
MuonH. The production launch requires compiled CUDA memory statistics and buffer
assignments to bound the complete device footprint at or below 72 GB. Compiler HLO and buffer assignments are enabled for that run.

The 1,000-update limit is a safety cap. The 20-hour wall-clock budget terminates
training, reserving 40 minutes for finalization and 10 minutes for an update.
After a 30-step linear warmup the base optimizer stays at peak LR. A rank-zero
monotonic-clock controller broadcasts an update multiplier that decreases
linearly from 1 to 0.1 over the final 15% of usable training time. With these
reserves, usable time is 19h10m, cooldown starts at 16h17m30s and lasts 2h52m30s.
Optax schedules apply the multiplier to all three optimizer groups before their
updates, preserving MuonH/AdamH norm projections and moment calculations.
Only remaining wall time triggers cooldown; step 850 does not trigger it.
Completion checks the final applied multiplier near the minimum for a wall-clock
stop. Reaching the safety cap earlier is valid at the then-current LR and never
forces cooldown. Telemetry logs the scale as `optim/learning_rate_scale`.
These settings are experiment declarations,
not restrictions on other schedule, precision or special-token choices.

Source revisions are recorded in [the curriculum](../task_curation/exp5_curriculum.py).
The science inputs are `rqa`, `so`, `syn_mcq` and `vendor` at revision
`1583049c1df02b418e07d14047eb37db9df46eba`; V7's existing publisher and native
consumer receipts qualify their retained population.

The curriculum owns a 200-prompt holdout removed from its training pools. At a
64-group batch, its mixture derives 13 agentic and 51 nonagentic groups.
Validation requires at least 180 scored prompts for best-checkpoint selection.
The launch verifies the pool producer and file hashes before using those pools.
The resolved plan and the graph bind the same pool artifact and curriculum.
The best-checkpoint threshold derives from that curriculum's holdout size.

```bash
uv run python -m experiments.post_training.long_loop.exp5 \
  --round experiments/post_training/long_loop/rounds/exp5-setup.yaml \
  --source-revision "$(git rev-parse HEAD)" \
  --version 2026.11.21.113 --stage sft \
  --validation-output /tmp/long-loop-exp5-setup
```

This command validates the SFT graph, writes local receipts, and submits no jobs.
After SFT finishes, continue through the same entrypoint with an explicit
certification seal and a fresh RL version. Set `COMPLETION_SEAL_URI` to the
published `seal.json` URI from the finalization certifier and
`COMPLETION_SEAL_SHA256` to its reported SHA-256:

```bash
uv run python -m experiments.post_training.long_loop.exp5 \
  --round experiments/post_training/long_loop/rounds/exp5-setup.yaml \
  --source-revision "$(git rev-parse HEAD)" \
  --version 2026.11.22.114 --stage rl \
  --completed-sft-record "$COMPLETION_SEAL_URI" \
  --completed-sft-sha256 "$COMPLETION_SEAL_SHA256" \
  --validation-output /tmp/long-loop-exp5-rl
```

The seal pins the native checkpoint, completion receipt and HF verification
receipt. The graph adopts the trained SFT run and its HF handoff, resolving
the existing `hf/step-N` export under the certified run root, where `N` is
the sealed checkpoint step. It does not repeat SFT or copy its weights.
Changed receipts or model metadata fail before launch. The full curriculum
and OpenResearcher admission must also pass; a certificate does not release
the data hold. Add `--run` only after the integrated plan is approved.

The US-EAST-08A CPU data producer is
`experiments.post_training.long_loop.exp5_data` at version `2026.11.20.1`.
Its output prefix is `OUTPUT_PREFIX` in [the producer](exp5_data.py); its
`long-loop-exp5/rl-pool-nonagentic/2026.11.20.1/manifest.json` pins the unchanged
Exp4 input, IFBench and HARDMath files and records all source counts. It processes all 702 raw
HARDMath rows with the base converter, retaining 684 tasks and preserving the
18 invalid-reference rejection receipts. It applies no difficulty filter and
does not select the RHS variant.

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

A recovery round pins its completed SFT and handoff artifacts and names its curriculum
pools and retrieval service explicitly. It never rebuilds SFT. The CPU launch preflight
also constructs the serving agent with its bound model limits: the per-turn output cap
must remain separate from the total trajectory budget and fit the compaction reserve.

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

### Mochi optimizer settings and open SFT follow-up

[PR #11640](https://github.com/marin-community/marin/pull/11640) is carried as
`specs/mochi-optimizer.yaml`, a standalone legacy Grug `OptimizerConfig` preset.
It selects corrected stacked-expert grouping, the hero 4-D Newton–Schulz backend,
`expert_normalization: all_experts`, `global_grad_norm: 1.0`, and `use_syrk: true`.
It retains the source PR's Snowball beta1, beta2, epsilon and learning-rate schedule.
To opt in, replace the `optimizer` object in a copy of a legacy Grug recipe with
this document and update that recipe's SHA-256 in its copied run spec. No existing
run spec selects this preset. `specs/sft.yaml`, `specs/round1.yaml` and
`sft-smoke.yaml` retain their current bytes and pins.

The native `snowball_h100` profile keeps Adam; `snowball_exp3` keeps its separate
hero optimizer. The Mochi RL rounds adopt pinned HF exports and do not run SFT.
The generic `grug_moe_muonh_v1` optimizer now uses the corrected grouping from
[PR #11639](https://github.com/marin-community/marin/pull/11639), including for
historical Grug recipes loaded with this source. Reproduce historical Grug runs
with their original pinned source revision; active run source pins are unchanged.

Open follow-up: the authors of #11639 and #11640 report two runs with experts in
MuonH on the 64-H100 data8 layout that did not finish their first step. The cause
is unresolved. The preset is not qualified for that 64-H100 data8 layout; diagnose
the first-step stall and confirm the SYRK backend on H100 before at-scale use.
No GPU job or smoke was launched for this integration.

### Starting RL from a published HF export

An `adopted_sft` file with `kind: hf_export` names an HF repository, its full
revision, the in-region copy's URI, and the size and SHA-256 of every object in that
copy. `specs/adopted-sft-20261005.yaml` pins the 2026-10-05 SFT release mirrored to
RNO2A. The adoption stage hashes every object, requires the exact object set, checks
the model geometry against the SFT recipe, the expert tensors against the declared
schema (1 = stacked, 2 = split), `tokenizer.json` against the loop tokenizer, and both
EOS tokens. RL then reads the copy in place; there is no export stage. Use
`--through rl`: this form has no native SFT run evidence for the archive.

## Evaluation and archive evidence

The loop sets `run.finalization: any_terminal`: once the RL child is terminal,
the launcher exports the trainer-format checkpoint selected by validation-reward
EMA after a completed save. This applies if RL failed, was cancelled, or was
terminated after a stall. Finalization does not convert a failed or killed RL
child into a successful graph: the coordinator fails after native finalization,
including when no checkpoint exists. Committed checkpoint/export files and the
native failure receipt remain available for a separate recovery. The ordinary
evaluation, report and archive graph runs after successful training and export.
Export failures also propagate; they are not treated as a missing checkpoint.
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
Workspace-state agent timeouts run the final-state verifier. A completed
verdict, including zero, keeps its reward and model-token evidence. Verification
errors and masked rows are absent from `avg_score` and `pass_at_1` denominators;
`num_scored` and `avg_verifier_score` expose coverage for each domain.
Missing or failed verification remains masked. The turn limit stays at128 and
best-checkpoint selection requires at least 90% of the expected validation
tasks scored (108 of 120 for this shape). An ineligible validation warns
loudly, is recorded with its reason in the selection receipts, and is skipped;
it never silently cancels best selection and never aborts training. When no
validation is eligible, export falls back to the final checkpoint with a
durable warning; when an eligible validation exists, the best-by-validation
checkpoint is required and its absence fails closed.


New Snowball and Mochi RL rounds share `RL_TRAINING_BUDGET` in
`execution_policy.py`: 43,200 seconds of learning and a 1,800-second drain
margin. The builders bind this clock even when a round omits the callback,
and reject conflicting learning or drain settings before allocation. The V5
Exp3 and Exp4 DSpark YAMLs also record these canonical values explicitly so
the authored plan shows its budget. Changing the shared constant requires
updating those declarations; stale values fail validation. Smoke runs retain their step
and outer-job caps and resolve the same training clock.

The clock starts at the first optimizer completion and stops at a completed-step
boundary after the budget expires. Startup time and the drain margin do not
reduce the learning clock; no minimum-update requirement extends it. An in-flight
step may finish before finalization. In full production rounds, the task and
coordinator timeouts must each exceed the learning budget plus drain margin.
Outer deadlines remain explicit in
`rl.execution.job_timeout_seconds`, `evaluation.execution.job_timeout_seconds`,
`execution.coordinator`, or `execution.task_timeouts` for SFT/export/eval-inputs/archive.
Frozen recovery plans and native resume accounting retain their recorded
budget; the lost-time deadline-extension rule is unchanged. Updating the
source constant does not change a running job.
Long-loop round specs, serving presets and launch defaults allow all NVLink
racks. Do not add rack exclusions. Existing jobs keep their submitted placement
constraints and must not be restarted just to change rack eligibility.

All new SFT rounds, including one-epoch runs, read the 20-hour
`SFT_TRAINING_DURATION` from the same module. The standard finalization and update
reserves are shared there too. Their YAML can specify reserves but does not repeat
the duration. The shared SFT builder and the SFT stage deadline use that constant.

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

A round can bind a content-pinned RL stage patch with `rl_recipe_patch: {path, sha256}`.
An adopted RL-only round uses `rl.recipe_patch` with the same `FilePin` fields.
The runner verifies the bytes and merges the typed patch before deriving
curriculum-owned validation settings. The graph builder then validates the
complete `SkyRLRecipe`. Model, curriculum, SFT and
canonical Micro210 bindings remain round inputs.

`specs/exp6-rl.yaml` sets a 131,072-token request window, 98,304 total response
tokens and 24,576 output tokens per turn. `context_budget.max_new_tokens_per_trajectory`
sets the total response cap separately from `max_new_tokens_per_turn`; prompt
capacity is the request window minus the total response cap. Without an explicit
trajectory cap, the effective total response cap equals `max_new_tokens_per_turn`.
Exp6 admits initial prompts up to 32,768 tokens. Pi's compaction reserve keeps
24,576 tokens free for the next reply within the request window. Its RL
batch remains 64 prompts times eight samples, with gradient clip 0.5. Its MuonH
optimizer follows Mochi's RL routing: main learning rate 1e-5, Adam fallback
learning rate 2.307692307692308e-6, and CPU storage for MuonH momentum. Adam
fallback state stays on the accelerator; AdamW CPU optimizer offload is disabled.
The qualified trainer uses four eight-H100 hosts with TP1/PP2/CP4/EP16/ETP1
(attention DP4, expert DP1). Declare this geometry in the round's role plan and
Megatron geometry; the stage patch does not allocate hosts. Policy and reference
use `all_gather` for context communication, preserving sliding-window attention
with Snowball's five KV heads. Output projection uses 1,024-token chunks.
The patch enables native warm-state sizing at 85,557 and 131,072 tokens before
weight publication or rollouts, with a required 10% headroom on every rank.
Qualification run `/benfeuer/mochi-exp6-muonh-memory-cp4-ep16-20261123-151`
passed on all 32 ranks at both lengths; minimum headroom at 131,072 tokens was
13.01 GB and peak reserved memory was 54.37 GB. Two retained-batch updates had
zero replica checksum mismatches and rank-median learner durations of 164 and
170 seconds for the two updates. These measurements used Exp5 SFT527 as
an architecture control; the Exp6 model and curriculum remain launch inputs.
Before a full run, use `trainer_memory_probe` to qualify the retained 64-by-eight
batch after optimizer state is initialized. Require the native updates to apply
and the warm sizing checks to leave at least 10% memory headroom, as described
under [trainer memory sizing](#trainer-memory-sizing-and-chunked-output-projection).
Bind this patch from the Exp6 round after its model and data are ready.

`sft.first_collective_call_timeout` sets XLA's first-call rendezvous termination
timeout in seconds. It allows cold-start skew between local GPUs without changing
the training computation or the separate NCCL operation watchdog. Exp5 uses
600 seconds after a 32-rank native-update qualification found matching compiled
programs and correctly sharded inputs; this is a timeout mitigation, not a proven
root-cause fix for the earlier first-step failure.

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

### Mochi priority evaluation and rolling route quotas

The prepared `rounds/mochi-throughput-train-rolling.yaml` round uses
`eval_drain: priority` and `batch_policy: rolling`. It is a no-run configuration;
submission requires explicit approval. Route quotas are derived from the
curriculum, giving 22 agentic and 42 nonagentic groups at batch size 64.
Each batch takes the earliest committed groups separately within each route.
A slow lease does not own a slot in a particular batch; a missing route must
still supply its quota. Generation capacity reserves the same route shares
across the staleness lookahead window.

Groups outside the configured policy staleness window are discarded and their
prompts returned to the existing retry loader. Commit-order selection favors
faster groups within a route, so exact route quotas do not guarantee the source
mixture inside a route. Group disposition events retain the source label for
measuring this bias. No stale or discarded group is scored as zero or trained.
Checkpointed rollout state records its batch policy. Resume requires the same
policy; a rolling run refuses full-batch state, including older state without a
rolling marker. Weight-only initialization is separate from resuming rollout
state.

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

### Mochi Micro210 evaluation

Launch Mochi evaluations through `experiments.evaluation.micro210` with the
`--model-uri`, `--model-manifest-sha256`, `--tokenizer-uri`,
`--tokenizer-revision` and
`--serving-profile mochi-gb200-cw-us-east-08a`. This registered profile selects
`export-manifest.json`, GB200 DP16/EP16 placement in CW08, the RunAI loader,
InfiniBand transport and a 4-MiB NCCL buffer. Compilation and PIECEWISE CUDA
graphs stay enabled. Omitting the profile keeps the frozen H100 allocation
used by other Micro210 runs. The policy is `2026.11.20-together`; the tokenizer
is `marin-community/marin-tokenizer-tools` at
`3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09`.

For a native RL checkpoint converted by `cloud.iris.export_hf_checkpoint`, also
pass `--model-manifest-filename .marinskyrl-model-manifest.json` and pin that
file's SHA-256. This selects the model's completion metadata, without changing
the frozen evaluation protocol or serving geometry. Omit the option for the
profile's original SFT `export-manifest.json` format. The Mochi profile excludes
the faulty `dh1-r136-us-east-08a` NVLink domain.

The standard Micro210 entrypoint and long-loop evaluation stages run the shared
preflight before allocating GPUs.
An isolated CPU task in the run's cluster checks policy and prepared-input
readability with only explicitly forwarded S3, tokenizer and task credentials,
then constructs and closes the real rollout worker without inference engines.
S3 and tokenizer credentials come from the submitting process’s named secret
files or environment. The launcher resolves the policy’s versioned judge
credential as described in [Micro210 judge policy](../../evaluation/README.md#micro210-judge-policy).
Installed-agent settings are validated against the runtime schema; the judge
connection remains in `environment.task_sessions.remote_judge`.

The base/SFT evaluation source using the earlier frozen MiniMax policy remains
on Git branch `benfeuer/mochi-micro210-gate2-frozen` (commit `da0663fbea`).
Current evaluations use the versioned Together API judge policy.

### Fresh Exp3 RL after the Socket failure

`exp3_rl_restart` adopts the sealed `.51` SFT export and data inputs without
rerunning SFT. Supply the hash-pinned predecessor plan and launch document as one
reference JSON object. The fresh lineage removes forced Socket transport, uses
the canonical RL learning budget, saves native checkpoints every two hours and at
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
the published BF16 SkyRL wheel `marin-vllm-gpu-20261118-47a8e415bff1`, pinned
in the `vllm` extra in [SkyRL dependencies](../../../vendor/skyrl/pyproject.toml)
from `config/external/vllm/gpu.toml`.
These are configuration ports. CPU validation does not establish GPU fit, serving readiness or throughput.

The one-rack round excludes `dh1-r136-us-east-08a` through
`rl.execution.excluded_nvlink_domains`. The launch document carries these domains
as required `nvlink.domain NE` Iris constraints. The Kubernetes backend translates
them to required node affinity on `ds.coreweave.com/nvlink.domain`. The target
controller must run this backend support before submitting an excluded-domain
job; an older controller does not enforce this constraint. NVLink role locality
alone does not exclude a rack.

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/mochi-one-rack.yaml \
  --through rl --version 2026.11.19.201 \
  --source-revision "$(git rev-parse HEAD)" \
  --validation-output /tmp/mochi-round-validation
```

To continue an adopted RL round into canonical evaluation, declare
`evaluation: frozen-micro210`, an `evaluation_profile` from the registered
Micro210 serving profiles, and `rl.export_hf: true`; use `--through eval`.
Set `rl.recipe.trainer.hf_export_checkpoint: best` to export the eligible
validation winner. `rl.retained_checkpoints: 1` retains one rolling resume
checkpoint while the trainer protects the selected best and final checkpoints
in permanent storage. The graph waits for the selected HF export before
evaluating it. The initial model receives a vocabulary check before RL; the
future selected model receives its own CPU gate after export.

Set `evaluation_checkpoints: best_and_final` to evaluate the best policy first,
then the final native state if it differs. Identical checkpoints reuse the best
evaluation receipt. Final export derives its geometry and production placement
from the original training launch. Each evaluation commits before its separate
preservation stage copies the temporary trial evidence to permanent regional
storage, so a copy retry does not rerun the evaluation. `coverage-ready.json`
records scored coverage, pass@1, per-benchmark counts and a zero-score ledger;
the #10197 seal still requires agent trace review and at least 90% valid trials.

The command validates without submitting. Normal runs use the canonical 12-hour learning
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

### Speculative decoding drafts

Speculative decoding is off by default in every round. To serve a frozen draft
during rollout, set `rl.draft` to a name registered in `round_runner.DRAFTS`; an
unknown name fails. The runner injects the draft into
`rl.recipe.generator.speculative_decoding` (an explicit block in the recipe
conflicts with `rl.draft` and fails), and the resolved plan records a
`draft_binding` with the draft's source, revision and the initializer its
weights were trained against. Long-loop rounds start from round-specific SFT
checkpoints — for example the Exp5 SFT step-527 handoff — while the registered
DSpark draft was trained against the pinned 09.21 Datakit Snowball SFT
initializer, so that binding is unmeasured on long-loop geometry until a
matched plain-versus-draft gate passes there (median update time, rollout-group
p95, acceptance length and fresh-policy KL on the same pinned data). No round
may enable a draft by default, and no running round changes its recipe.

The DSpark draft shares the rollout vLLM engines and needs dense-draft support
with an expert-parallel target from the pinned serving wheel
`marin-vllm-gpu-20261118-47a8e415bff1`. Keep compilation and CUDA graphs
enabled; verify the serving pin stays consistent with the SkyRL lock
(`tests/test_serving_pin.py`).

### OpenResearcher pools on adopted RL rounds

An adopted RL round whose pools carry the OpenResearcher route must bind
`rl.openresearcher_service` (dataset id, revision, regional index URI and the
pinned Parquet SHA-256). The runner cross-checks the binding against the
pools manifest's `openresearcher` admission, applies the service's task-machine
contract, rides the judge credential source, and starts an attempt-owned
retrieval service around the RL run. A binding without the route — or the
route without a binding — fails; the resolved plan records the binding.

### Adopting a certified SFT export

An adopted RL round binds its model one of two ways, never both. The default
pins `export_manifest_sha256` and takes the checkpoint step from that export
manifest. A certified export instead pins `model_manifest_sha256` (the
`.marinskyrl-model-manifest.json` staging manifest beside the export) and
`certification_seal` (path and SHA-256 of the certification seal); the runner
verifies the seal is COMPLETE, certifies exactly the declared checkpoint step
and completed updates, matches the native checkpoint, HF export path and
receipts against it, and refuses temporary or inconsistent outputs. This
route exists because an export's `export-manifest.json` can predate the
certified weights — the step evidence is the seal, never that manifest.

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

## Serving load measurements

`experiments.post_training.long_loop.throughput --spec <regional-json-uri>` runs
qualification or a load sweep inside an Iris GPU gang. It uses the shared Marin
`VllmBackend`, the pinned GPU wheel and native Iris/vLLM topology. Qualification
requires successful streaming requests and captured CUDA graphs in each task's
retained log. It does not enable eager execution.

The JSON spec declares `mode` (`qualify` or `benchmark`), `source_revision`,
`model` (`ModelSpec` fields), `engine` (`VllmEngineConfig` fields other than the
pinned launcher/source/release), `geometry` (`ServingGeometry` fields), and
`output_uri`. A benchmark also declares `workloads`, `seeds`, `concurrencies`,
`warmup_seconds`, `measurement_seconds` and `request_timeout_seconds`.
Synthetic workloads declare `kind: synthetic`, a `name`, `context_repetitions`
and `output_tokens`. Replay workloads declare `kind: replay`, a `name`, an
immutable JSONL `uri` and the `request_digest({"requests": rows})` SHA-256.
Replay rows contain `identity`, `path` and `body` from `LoadRequest`.

`marin.inference.load_workloads` builds the same synthetic text for both models
and extracts assistant-turn prompt prefixes from retained native Micro210 rows.
The replay excludes the response being regenerated. Tool-dependent prefixes
require recorded tool declarations. Preserve a source inventory and its hashes
beside the replay; keep prompts and results in private object storage.

`marin.inference.load_test` keeps a closed-loop client load active through warmup
and the measurement window. It records final token usage, request TTFT, TPOT,
decode rate, errors, server token-counter deltas, queue time, KV use and prefix
hits. Server counters define window throughput. Request-cohort throughput
includes drain and is reported separately. Prompt counters include cache hits;
they are not isolated prefill compute throughput. Missing/reset counters remain
unavailable, and a failed request invalidates the run. Retain raw per-request
timings and labeled metric samples to audit normalization across DP ranks.

Submit through Iris at production priority, with explicit GPU count, rack
locality and job deadline. Prepare large model snapshots with a CPU job in the
same region before reserving accelerators. Record every submission ID and hand
long jobs to the task's watchdog.

## Bounded Mochi decode arms

The `throughput` worker measures serving without trainer or task-environment work.
The checked-in `specs/mochi-decode-seqs16.json` control uses the Mochi rollout
engine's model, TP1/DP12/EP12, BF16, cache, logprob, compiler and CUDA graph settings.
The `seqs32` spec changes only sequence capacity. The next queued specs are
`mochi-decode-seqs32-kv85.json` (GPU memory utilization 0.75 to 0.85) and
`mochi-decode-seqs32-kv85-batch16384.json` (batched tokens 8192 to 16384). Each
changes one engine parameter from its predecessor and keeps offered concurrency
384. All use a 25-minute server-counter window after warm-up, plus retained
streaming request timings. The `seqs32-kv85-shared8` spec then changes prompt
group size from one to eight, retaining 256 request identities across 32 shared
prompts. The `seqs32-kv85-shared8-short` spec changes only context repetitions
from 64 to one. These isolate group prefix reuse and prompt length before another
sequence-capacity change. The loader streams
regional weights for startup; throughput excludes loading and request drain.

Submit from the clean, landed long-loop commit:

```bash
uv run iris --cluster=marin job run \
  --job-name mochi-throughput-decode-control-20261121a \
  --target-cluster cw-us-east-08a --priority production \
  --gpu GB200x4 --replicas 3 --cpu 120 --memory 900GB --disk 2TB \
  --enable-extra-resources --extra cpu --timeout 10800 --max-retries 0 --no-wait \
  --exclude-nvlink-domain dh1-r136-us-east-08a \
  -e NCCL_NET IB -e NCCL_IB_DISABLE 0 -e NCCL_IB_HCA ibp -e NCCL_BUFFSIZE 4194304 \
  -e PYTORCH_CUDA_ALLOC_CONF expandable_segments:True \
  -- uv run --no-sync python -m experiments.post_training.long_loop.throughput \
  --spec experiments/post_training/long_loop/specs/mochi-decode-seqs16.json \
  --source-revision "$(git rev-parse HEAD)"
```

Source revision is a separate required argument so a committed spec does not embed
its own commit hash. Use a fresh job name and output URI for every arm. Stop a
measured arm with `iris-cancel-own` if it is still running after publishing its
window. Compare generated tokens/s per engine and GPU, and request decode rate
at actual concurrency. Do not infer serving capacity from consumed training tokens.

### Trainer memory sizing and chunked output projection

For a qualified Megatron PPO recipe, set
`trainer.policy.megatron_config.memory_profile.enabled: true` before launch.
Declare `max_sequence_length`, `valid_token_counts` (which may be empty), and
`minimum_headroom_fraction` explicitly in the same mapping; use `0.1` for a 10%
device-memory reserve. Omitting an enabled check's capacity or reserve fails preflight.
The check runs on every policy rank before initial weight publication and rollout
admission. It initializes optimizer state, then runs forward, backward and optimizer steps
at each `memory_profile.valid_token_counts` value and at the explicit
`memory_profile.max_sequence_length` (262,144 for Exp5). Include observed long trajectories, such as
85,557 valid tokens, as well as the configured maximum. The scheduler fills the
pipeline's activation slots with native microbatches. The trainer-only
`trainer_memory_probe` requires `--max-sequence-length` explicitly; use 131,072
for the Exp6 context. The same qualification run measures 85,557 valid tokens
and the configured maximum with warm optimizer state. This does not change a
running trainer's configuration.
For router-replay policies the synthetic response carries complete, balanced
expert choices through the normal replay controller; receipts name that routing
mode. This tests capacity, not a bound on arbitrary expert-load imbalance.

Each rank writes `trainer-sizing/rank-<rank>.json` under the run's export root.
The receipt includes allocator peaks, duration and an estimate of device headroom
that includes memory outside the allocator at the interval boundaries. Transient
allocations outside PyTorch are not measured peaks. An OOM or headroom below
`minimum_headroom_fraction` fails startup. Model, optimizer, scheduler and RNG
state are restored before training; sizing updates do not become learning steps.

`output_projection_chunk_size` bounds the vocabulary projection and its backward,
while `logprob_chunk_size` alone bounds only softmax temporaries. The chunked
projection uses Megatron's output processor and native weight-gradient
accumulation. It retains hidden states and recomputes one token chunk in backward,
returning scalar logprobs and entropy through the existing CP reconstruction.
This path currently requires TP1 PPO without sequence parallelism, FTPO or top-K
distillation. Both options are disabled by default. Qualify values, gradients,
restoration, peak memory and throughput on the intended CUDA geometry before
enabling them or removing optimizer offload in a launch recipe.

### OpenResearcher with hosted retrieval

Exp5 binds the same service contract for the OpenResearcher partition of its
eleven-source pools. Its round YAML declares the qualified source and a regional
index URI. Only that source gains the network-denied Daytona actor; other
sources, turn limits, judge settings and verifier resources remain unchanged.
The run-owned service is ready before RL GPU allocation. A regional CPU
preflight lowers one task per training input and one per validation source,
then checks the actual machine factories and harnesses without provisioning
sandboxes or starting inference. Its receipt records source, task ID, phase and
machine bindings. The isolated CPU child receives the parent's W&B, Hugging Face,
object-store, Daytona and judge secrets. An unsupported binding or missing
required credential fails the parent before GPU allocation. Daytona target and
API URL remain nonsecret environment settings so native clients read the
configured route. CPU worker construction applies the same resolved task
environment as the GPU launcher, including the declared judge model; it does
not select another judge. The receipt records the model and environment names.

Named backend credentials resolve file-first through Rigging in the launcher,
remote judge, installed-agent connections and both task-machine phases. The CPU
gate validates every configured train/eval credential and records only its name.
Campaigns also require their declared storage, Hugging Face and W&B secrets.
Worker construction keeps those secrets in their read-only files and projects
only authored runtime settings into the environment, matching Ray workers.

The full-index OpenResearcher admission pin runs on LOCAL/Iris. Daytona RL uses
`open-athena/openresearcher-openresearcher-dataset-service-20261121-v1` at
`0c4f39584d7f0121e97f6ad4f4761749add56fef`. This private variant requires the
frozen BM25 service. It retains all 283 tasks and the native grader; only its
actor image and importer revision differ from the full-index pin.

A one-source NVL72 run spec selects the private variant with these input fields:

```yaml
data_uri: []
data_sha256: 346e892a7052d1f93ee31ae208100adccee06ff1918344efe2974fabc331226c
openresearcher_service:
  dataset_id: open-athena/openresearcher-openresearcher-dataset-service-20261121-v1
  dataset_revision: 0c4f39584d7f0121e97f6ad4f4761749add56fef
  parquet_sha256: 346e892a7052d1f93ee31ae208100adccee06ff1918344efe2974fabc331226c
  index_uri: s3://marin-us-east-08a/marin/rl-data/openresearcher/bm25-index/2026.11.15.1
```

Keep the other model, tokenizer, artifact-version and role-plan fields explicit.
The input builder downloads the immutable private pin as an artifact dependency.
Its Parquet digest is checked after download and before service launch. Config
rendering does not read an input artifact that has yet to run. Set the index URI to the run's own
region; the service refuses a bucket in another region.

The coordinator starts one Iris CPU child per run attempt with 8 CPUs, 32 GB RAM
and 128 GB disk. It verifies every frozen index file before private discovery.
The ready endpoint enters `environment.task_sessions.retrieval_service` at
launch. Each thin actor keeps external networking denied and uses a private
Shellbox relay for `search` and `get_document`. Grading uses the original script
in its separate verifier machine. The child remains alive through the run and
terminates during cleanup. No endpoint or credential is stored in the dataset.

The Daytona verifier uses a machine-local HTTP relay for its native judge
requests. The owning host forwards the original payload with its HTTP client;
the upstream judge credential stays on the host. This avoids the provider's
rejection of the native Python `urllib` client without changing grading rules.

This source path does not require main runtime adoption or a consumer-gate
profile. Message-trace source qualification and exact-token RL training are
separate checks; a source rollout does not establish an optimizer update.

### Mochi Track 1 preparation

`rounds/mochi-throughput-train-track1.yaml` aligns checkpoint, evaluation and
best selection every three steps and evaluates the initial policy through the
evaluation callback. Generation and agent budgets match the 30-minute recipe.
For draining modes, the trainer pauses new rollout admission after acquiring an
evaluation step's batch, before learner work. Every round explicitly selects
`trainer.rollout_buffer.eval_drain`: `wait` lets submitted groups finish, while
`abort` revokes their leases, cancels generation and waits for worker sandbox
cleanup acknowledgement. Aborted groups never enter a training batch or its score
metrics; their prompts are re-queued without spending a staleness retry. Late RPC
arrivals and commits under revoked leases are rejected. Runtime events record the
strategy and aborted/re-queued group count. Ordinary trajectory timeouts still
run the verifier and retain its score.
Pausing before batch acquisition could prevent rejected groups from being replaced.

`priority` leaves training rollout dispatch and in-flight groups active. Evaluation shares
the same rollout workers and vLLM engines, using request priority `-1` versus
training's `0` (lower values run first). Recipe composition derives
`generator.engine_init_kwargs.scheduling_policy: priority` from this mode; a
conflicting scheduler or colocated engines fails validation. This changes queue
ordering, not the available KV memory or Harbor slots; running training requests
can still compete for engine resources.

After step-N weight sync, priority evaluation runs asynchronously with captured
step-N identities. Admission of the next learner batch waits for evaluation and
the deferred buffer policy publication to complete, so the consumed step-N batch
cannot be taken again. Weight sync also retains its evaluation fence. Best selection reuses the
saved step-N checkpoint rather than saving newer optimizer weights under an old
step. Keep checkpoint and evaluation cadence aligned when using best selection.
Evaluation failures propagate through the admission or sync fence before further
batch consumption or weight updates. Native timings retain `eval` at the evaluated
step and `eval_sync_wait` at the next learner step's fence.
The `.124` priority trial failed at batch admission before the fence was added.
The prepared `.125` round retains `.123`'s `abort` drain until the H100 comparison
qualifies priority mode. It uses 64 prompt groups with eight samples each, a
64-group policy minibatch and gradient clipping at 0.5. Learning rates retain
`.123`'s MuonH 1e-5 and Adam fallback 2.307692307692308e-6. The curriculum derives 22 agentic and
42 nonagentic groups. Staleness stays at three steps and the response/window
budgets stay at 98,304/131,072 tokens. Preparation does not authorize a launch.

For the bounded control-flow regression, `priority_drain_debug` uses the existing
regional Qwen3-0.6B checkpoint on two H100s (one policy GPU and one serving GPU)
in 02a at interactive priority. Its shared recipe is
`rounds/priority-drain-h100.yaml`; only the drain mode and run identity vary.
It runs eight steps with evaluation every two steps, no baseline or checkpoint
work, four GS2 training groups, and a 256-token response cap. The fixed 2,048-row
evaluation cohort extends each session through its size and low concurrency.
Train and eval both use full-support temperature-only sampling with `min_tokens: 0`;
forcing a minimum output length is incompatible with requested behavior logprobs.
Eight workers each admit two tasks; evaluation admits two tasks globally, while
the four-group training in-flight limit leaves workers available for eval.
The 64-step lookahead keeps background generation supplied during the eval fence.
These are control-flow settings, not a Mochi throughput or quality comparison.

```bash
uv run python -m experiments.post_training.long_loop.priority_drain_debug \
  --drain priority --version 2026.11.23.201 --source-revision "$SOURCE_REVISION" \
  --validation-output /tmp/priority-drain-validation
```

Run the CPU no-run preflight before adding `--run`. Use an immutable pre-fix
source for the negative control and the landed admission-fence source for the
fixed, `abort` and `wait` arms. Keep their combined allocation below 16 H100s,
retain eval/frozen-policy/admission evidence, and cancel each measured root tree.
No Mochi GB200 priority relaunch is authorized by this debug recipe.

For separate policy and rollout GPUs, the cadence snapshot overlaps frozen-policy
evaluation after weight sync. Best selection joins and reuses that snapshot before
publishing its pointer; step exit also joins on evaluation error or cancellation.
The next optimizer cannot mutate weights while their snapshot is running.
Colocated configurations retain sequential saving because checkpoint residency
management can sleep their rollout engines. Storage failures retain the existing
checkpoint failure metric and the last complete checkpoint.

`trainer.policy.megatron_config.checkpoint_plan_cache` explicitly enables cached
Direct S3 saving in the prepared Mochi `.122` round. It stays false by default
for other role plans, including Snowball, until their geometry is qualified.
With the option enabled, each policy actor retains the MCore save strategy and
its FullyParallel wrapper. MCore 0.18 metadata/local/central-plan caching and save
distribution caching stay in memory across destinations. Each save still creates
a fresh streaming writer and resolves the current tensor payloads. The coordinator
carries the BF16-to-FP32 master alias map with the plans and reapplies it before
finalizing a new checkpoint; cached logical metadata never receives an earlier
checkpoint's storage records.

Before distribution, each rank fingerprints tensor keys, shapes, dtypes, shard
coordinates, replica metadata and process-group membership. A change or failure
on any rank invalidates distribution, plans and aliases for all ranks. DCP's
collective error handling also covers fingerprint failures. Normal access-integrity
validation remains enabled. Four consecutive stable saves exercise reuse; key,
shape, dtype and one-rank-only changes exercise rebuilding. Both model-only and
optimizer restores must remain bit-identical at every saved destination.

`megatron_checkpoint_state` records materialization/setup and state construction.
`megatron_checkpoint_save` records the destination, rank, physical save index,
structure digest, enabled/invalidated/validated/reused cache state and separate
preprocess, fingerprint, distribution, key-translation, planning and writing
times. MCore first builds and then validates plans before reusing them. Compare
cold saves with later reused saves at the same geometry and report the maximum
across all policy ranks, rather than summing parallel durations. State construction
and access-integrity validation still run: planning time alone is not the full
checkpoint wall time. Production `.122` timing requires a separately authorized
training launch.


## Mochi response-cap cohort

`throughput --spec specs/mochi-context-cap-{16k,32k,64k,96k}.json` uses the same
native serving path as the decode benchmarks. Each arm issues a finite cohort of
80 Nemotron validation prompts and 300 uniformly sampled nonagentic training
prompts, eight paired seeds per prompt. The input producer is `context_workload`;
it verifies the complete pinned source hashes and publishes only public prompts
and answer-format metadata to regional storage. It does not change RL validation.

The four arms share TP1/BF16 weights and KV, EP16/DP16, a 131072-token window,
HBM fraction 0.90, sequence capacity 64 and batched tokens 8192. Only the response
cap changes. The window fits the longest 5944-token input plus the 98304-token cap.
Production 08A placement excludes r136. CUDA graphs and compilation stay enabled.

`cohort.json` retains each attempt's finish reason, token count and timing.
`responses/` retains complete text, tool calls and exact token IDs. A failed or
unfinished request is not a length stop: partial results include bounds over all
expected attempts and cannot establish the final truncation fraction. Metrics are
saved in short chunks; `progress.json` updates approximately every minute. Decode
rates over the complete cohort include declining concurrency at its tail; use the
metric chunks for saturated-window comparisons. Hard periodic-tail detection is
shared with training; it does not detect all circular reasoning. Normal source
answer extraction can accept raw text, so explicit final markers are also
reported. Neither signal is a correctness grade.

## Mochi turn deadlines and partial responses

The Mochi Exp3 and Exp4 round recipes set
`generator.engine_init_kwargs.request_timeout_seconds: 1770`. The asynchronous
vLLM wrapper ends an active decode through vLLM's abort API, reserving 30 seconds
inside the 1,800-second turn deadline for returning its exact partial response.
Tokens, behavior logprobs, routes and policy-step stamps are retained; no EOS
token is synthesized. The transport marks the turn `total_turn_timeout`, and
the rollout engine runs the task verifier on the retained response and refuses
another model turn. A request that generated no tokens is left to the outer
deadline and remains unavailable. The 2,280-second attempt limit is unchanged.

An archived `generated_tokens=0` with “Turn deadline expired before the first
response” means no completed response reached the rollout recorder. It does
not establish that the serving engine generated no tokens. Correlate the
session ID with `request_timing` and inspect engine queue metrics. SkyRL's
engine snapshots are in `telemetry_v1.marinskyrl` with `metric_source=vllm`;
standalone serving uses `telemetry_v1.vllm`. Unavailable rows have zero token
loss masks and are excluded from RLOO baselines. Retained generation samples
also include groups the learner rejects, so report their rates separately
from consumed-batch metrics.

Use a fresh round version after adopting this runtime change. Editing a recipe
does not change an already submitted run.

The shared clock-aware task client also requests partial output before its
nearest active execution deadline. After actual engine admission it reserves
up to 30 seconds for returning the response; when less than 60 seconds remains,
it reserves half that remaining time. This uses the remaining trial/turn/agent
budget, including time spent in earlier turns, rather than resetting a fresh
per-request deadline. It covers canonical evaluation even when the serving
profile has no `request_timeout_seconds` setting, without overriding the frozen
protocol. The client awaits the original response collector after the native
abort, retains its exact token metadata, and marks a returned native abort as
`total_turn_timeout` so the real verifier runs. Requests that generated no tokens remain unavailable;
caller cancellation remains cancellation.

## Mochi DSpark serving gate

`rounds/mochi-exp4-dspark-rl.yaml` prepares Exp4 RL from the regional Mochi SFT
V5 SFT step-2100 export and the unchanged interleaved two-layer 766M draft from
[PR12239](https://github.com/marin-community/marin/pull/12239). It keeps the
128-GB200 layout (48 trainer and 80 rollout GPUs), TP1/BF16, batch64×8,
priority drain, rolling route quotas, and the normalized Mochi context and
optimizer. The source inputs are the original Snowball Exp4 `.71` agentic and nonagentic
pools, pinned through the `exp4-recovery82-plan.json` record, parent curriculum
`long-loop/exp4-rl@2026.11.18:cde16ecb`. The preparer
`experiments.post_training.long_loop.exp4_control_data` preserves every original
serialized task and source entry, draws200 prompts with seed20261020, removes
them from training and publishes a new curriculum identity. The new holdout contains200 prompts and requires at least180 valid scores for
best-checkpoint selection. It replaces Snowball Exp4's original120/seed20261109
validation; that
split and the200 removed training prompts are known comparison differences.
The later `.22.2/c34a2771` pools are not used: they repair Reasoning Gym scorers,
use Nemotron recipe2 and exclude191Codeio rows. The derived route quotas are
22 agentic and42 nonagentic groups.
Four proposals use vLLM memory fraction0.775 and the Nov24 ShortConv-fixed
wheel. The draft was fitted to step400407; this experiment deliberately uses it
as-is on V5 step2100, without refitting or changing the initializer.

Prepare the complete RL/best-and-distinct-final evaluation graph without
submitting it:

```sh
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/mochi-exp4-dspark-rl.yaml \
  --version 2026.11.25.131 --source-revision <landed-sha> \
  --validation-output /tmp/mochi-exp4-dspark-validation --through eval
```

Launch readiness also requires a matched sampled-token logprob gate. The
PR's 37% reduction in warm-update time was measured on a different 60-GB200
workload and is not a speed prediction for this round. Its trainer/rollout
mismatch means do not replace the sampled-prefix check below.

`experiments.post_training.mochi_dspark_gate` runs one TP1/BF16 EP16 gang.
It keeps the EAGLE3 harness's frozen probability, routed-expert and sampled
distribution limits. Independent plain V5 decoding supplies the checkpoint-matched
reference, followed by DSpark and a fresh plain V5 teacher rescore. The teacher
rescores the exact sampled-token prefixes from both arms and compares their
returned token logprobs under the unchanged numerical limits. Any failure stops the gate before performance measurement.
Use `--purpose qualification` to stop after the plain, DSpark and independent
plain-rescore checks; `--purpose benchmark` also measures performance after
all correctness checks pass. The spec pins the flat draft's verified-copy
manifest and the runtime script. The script uses the published Nov24 wheel
without source patches, in an isolated environment. Run the serving-only gate
inside its registered 08a Iris gang with the published spec and landed source:

```sh
python -m experiments.post_training.mochi_dspark_gate \
  --spec <regional-gate-spec-uri> --source-revision <landed-sha> \
  --purpose qualification
```

For a failed plain control, `--purpose plain-batching` runs only the plain
server. It scores the frozen prefixes twice at each `(requested rank, request
concurrency)` pair: `(automatic, 16)`, `(automatic, 1)`, `(0, 1)`, `(1, 1)` and
`(0, 16)`. Explicit ranks use `X-data-parallel-rank`. Engine settings
and numerical limits stay unchanged. Failed comparisons are retained so the
remaining diagnostic conditions can finish; completion is not qualification
and never writes `correctness-passed.json`. Request concurrency does not fix
the realized GPU batch shape, and successive conditions share a cache whose
eviction is not controlled. Use serving-only INTERACTIVE or BATCH jobs for
this investigation and retain all task completion and graph-capture evidence.
The diagnostic spec must set `plain_flash_attn_version` explicitly: `null`
keeps automatic FlashAttention selection, and `2` forces FA2 through vLLM's
`--attention-config`. This isolates attention version from the other settings
in `VLLM_BATCH_INVARIANT`. Qualification and benchmark paths keep automatic
selection.

The diagnostic spec must set `plain_prefix_caching` explicitly. Use `false`
for a separate cache-disabled arm with all other settings unchanged; the
qualification and benchmark paths continue to enable prefix caching.

The inference-stats callback records cumulative draft/accepted-token counts,
acceptance rate and mean acceptance length at each training step. Use per-engine
counter differences between updates for each update's acceptance; account for
process resets and do not sum forwarded counter snapshots. A zero draft-token
denominator leaves acceptance unavailable. Full `Training step completed` durations and native
`phase_duration_seconds` observations record update wall time, including admission
and synchronization. Optimizer time alone is not the full update time. No training
acceptance or update-time result exists before the user authorizes RL.

After correctness passes, `mochi_decode_window` measures exact token arrivals
after every rank-pinned stream has prefetched. Each plain/DSpark cell requires
16, 32 or 48 running streams on every engine, no KV preemptions and a full common
measurement window. Natural EOS is preserved; early completion invalidates the
cell. Report tokens per verification round, per-request decode speed ratios and
TPOT p50/p95 from valid matched cells. These windows exclude initial prefill and
measure serving in isolation. A training-pair comparison needs separate approval.

## Campaign stall stop

The user-directed campaign cutover enables `repeat_stall_batches: 3` in
`campaign/mochi131.yaml` for every newly prepared rollout. This intentionally changes
the campaign controls. Earlier running roots retain their submitted configurations.
An unset limit preserves the previous runtime behavior. Three identical completed
action/observation batches stop further generation; wait-only actions are exempt.
The ordinary final-state verifier still grades the episode, including an incomplete
solution that earns zero. Stopping never supplies a reward.

The existing consumed-rollout metrics emit `generate/repeat_stall_fired_count`,
`generate/repeat_stall_fired_fraction`, and
`generate/repeat_stall_stop_reason_coverage` each step with stop metadata. Fractions
count terminal trajectories once, rather than weighting multi-turn episodes by
turn count. The trainer's existing tracker and training telemetry mirror these
scalars to W&B and Grafana. Fired count and fraction report zero on a non-firing
step; coverage reports the availability of terminal stop reasons. These metrics
are omitted when the stop-reason field is absent.

Run the local campaign-config grading proof with:

```bash
uv run python -m experiments.post_training.long_loop.campaign_stall_e2e --output /tmp/campaign-stall-e2e.json
```

`campaign_pool_proof` is a CPU proof worker over the existing `campaign_graph` pool
dependency. It submits no training or GPU stage. In-region, it prepares one queued
mix with an agentic source, verifies every prepared rollout has the stall limit and
unchanged task/grader, and repeats the grading proof. Use a fresh pool version after
the cutover: rollout limits are baked into prepared pools, so an earlier pool proof
does not establish the enabled limit. Campaign RL continues through its existing
launch path. The campaign coordinator inbox records the landing commit and cutover.

## Polaris grader repair candidates

`polaris_release` derives a candidate from the retained Polaris release at
`9079512fe3d2d332b446b1b37deb9d2898f5461e`. It preserves all 53,167 task identities,
questions, references, runtime values and extra columns. It replaces the legacy
format-dependent grader and its package lock, records old and new task hashes,
and retains the original release metadata. A bad task hash, duplicate identity
or missing row fails the candidate build without publishing completed output.

Print the artifact plan without downloading the cohort or building an environment:

```bash
uv run python -m experiments.post_training.long_loop.polaris_release --version 2026.11.27.2
```

Build with `--run` inside an Iris CPU task in the target storage region, with
`MARIN_PREFIX` pointing to that region. The graph first builds the pinned grader
environment, then streams the retained cohort in batches of 64. It writes a new
candidate artifact and verifies uploaded file hashes. It does not publish a
Hugging Face release, alter the catalog, regrade a live run or launch training.
Failed artifacts require investigation before a fresh attempt; the command
does not automatically retry them.

A candidate needs three checks before release: its Linux grading guest,
regression controls and a fresh independent review. A development replay is not
that review. Existing ambiguous reference defects remain recorded without edits
or filters. Only a separately qualified immutable source pin may release campaign
holds; existing pools retain their original grader.

Candidate `2026.11.27.2` (policy v4) passed all three checks and is released as
`open-athena/polaris-project-polaris-dataset-53k-grader-v4-20261127-v1@8f1ecf9a2711b366253a6c76ffbad45a302d5013`.
Its `qualification.json` records the results: 47/47 guest controls, a 99-case
fresh replay, and a 51-case blinded review with no false rewards. The review found
four false zeros, all fail-closed. The campaign `source_pins.json` binds this
release. The two multi-target, partial-reference questions fail closed and stay
recorded as reference defects.

The v3 candidate recognizes explicitly delimited explanations, final proof displays,
signed changes, and assignments to the variable requested by the question. It
checks that a labelled reference agrees with the question's declared option before
removing the label. These rules never select an explanatory number by matching
the reference value.

Invalid reference data produces `infra_error` with `reward=None` when the grader
entrypoint runs, including with an empty answer file. Native submission grading
first applies the TaskSpec's empty-output policy: `empty_output=zero` can return
`submission_failure` with reward 0 before calling the grader. The separate
`grade_empty_in_sandbox` audit invokes the grader directly with an empty answer
file and leaves that submission policy unchanged.

The v4 candidate also recognizes the complete multiplication/subtraction digit-puzzle
grammar and verifies both equations and distinct digits 1–9. A complete assignment or
subtraction equation must determine exactly one valid full solution, independent of
the answer key. This preserves the original subtraction reference without accepting
unrelated equalities. Other prompt grammars use the scalar comparator. Multi-target
questions with an unlabelled partial reference still require a faithful target
binding; their known regressions remain strict expected failures and prevent admission.
