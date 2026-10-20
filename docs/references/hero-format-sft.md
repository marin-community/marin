# Hero checkpoint preparation and format SFT

The current Hero backend supports offline LR-aware checkpoint averaging,
output-only chat-token initialization, and LoRA plus compact vocabulary updates.
Format acquisition uses the source 16K configuration. A separate YaRN candidate
extends the merged format checkpoint to 65,536 positions.

These entrypoints prepare and evaluate new artifacts. They do not modify the
production Hero run. Full-model quality, throughput and serving parity require
separate measurements; CPU tests establish local implementation behavior.

Candidates use these names throughout the workflow:

| Candidate | Artifact |
| --- | --- |
| A | Latest permanent pretraining checkpoint in the source 16K regime. |
| B | Raw LR-aware average of the selected permanent checkpoints. |
| C | Selected A or B with five output chat-token columns initialized from donors. |
| D | Format-SFT result on C, exported as ordinary Hero weights. |
| E | D's weights with the 65K YaRN forward configuration. |

The JSON inputs follow their source schemas:
[CheckpointPin, LrSample and AverageConfig](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/averaging.py),
[CatalogSelection, SourceSpec and DataRecipe](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_data.py),
[Panel, CandidateEvaluation and GatePolicy](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_eval.py),
[NativePanelRequest](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_sample.py),
[FormatSftRecipe and LaunchEvidence](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_pipeline.py),
and [ContextExtensionConfig](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/context_extension.py).
Use measured evaluations and reviewed limits when filling these records.

## Checkpoint inventory and placement

Use `experiments.grug.moe_hero_ep.format_inventory` to inventory permanent
snapshots and pin their metadata, layout manifests, saved model and optimizer
configuration. It uses `hero_checkpoint_paths` for ancestry discovery and selects
only snapshots within the requested run's saved 16K regime. The inventory reads
metadata, not weight arrays.

```bash
uv run python -m experiments.grug.moe_hero_ep.format_inventory \
  artifacts/hero-format-sft/inventory \
  --run-id hero-ctx16k-step257k --checkpoint-count 3 \
  --source-revision <implementation-commit> \
  --tokenizer-revision <tokenizer-commit> \
  --tokenizer-revision-evidence <durable-evidence-reference>
```

Historical saved run configurations name the tokenizer without its revision.
Record the evidence connecting the source to the selected immutable tokenizer;
passing a revision string alone does not establish that connection. If historical
bytes cannot be established and the user accepts a revision assumption, retain
that explicit decision in the referenced record and mark historical bytes as
unverified.
`inventory.json.selected` contains the `CheckpointPin` records for
`AverageConfig.checkpoints`. Preserve access to the emitted `saved-config.json`
at each pin's `config_path` when moving execution to the verified worker.
`--run-id` defaults to `current_run.CURRENT_HERO_RUN_ID`; set it explicitly for
an immutable handoff. `--checkpoint-count` selects the latest compatible
permanent snapshots in that run's exact root.

Averaging runs on one CPU worker in the source storage region. All source
checkpoints, temporary storage and the output checkpoint must match its verified
physical placement. The production probe uses storage-service region metadata
and authenticated Kubernetes pod/node metadata, or Iris worker registration
metadata outside Kubernetes. Missing evidence or permissions stops execution
before weight reads. Do not substitute bucket names, cluster names or a
caller-supplied region for placement evidence. Some providers distinguish region
and availability-zone labels; resolve that distinction with provider evidence
before launching.

Select a CPU worker using [Iris Operations](https://github.com/marin-community/marin/blob/main/lib/iris/OPS.md) and the
[use-iris workflow](https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md). Use
`uv run iris --cluster=<cluster> rpc controller list-backends` to inspect the
target cluster before submission. `--cluster` selects the controller;
`--region`/`--zone` constrain scheduling using the cluster's labels. Those labels
can be coarser than a storage-service region, so the production placement probe
must still verify the worker and every storage path before averaging.

The default shared scratch bucket can be in a different region. Set
`AverageConfig.temporary_storage` explicitly to storage in the source region.
Averaging never mirrors the source checkpoints to another region.

## Averaging and donor preparation

Encode `averaging.AverageConfig` as JSON and run it on the verified worker:

```bash
uv run python -m experiments.grug.moe_hero_ep.averaging --config average.json
```

The request pins each checkpoint's metadata, manifest and saved configuration,
the tokenizer revision and its evidence, source revision, context length, and
logged LR samples. A logged Hero `global_step=g` describes completed update
`g+1`, which used scheduler index `g`. The implementation validates reconstructed
Muon and Adam schedules against several samples before integrating them.
Export W&B history for the source run with `global_step`, `optim/learning_rate`
(Muon) and `optim/adam_lr`. Convert each complete row into a `LrSample` with
`completed_update = global_step + 1`, `muon_lr = optim/learning_rate` and
`adam_lr = optim/adam_lr`; put those records in `AverageConfig.telemetry`.
Retain multiple samples spanning the selected checkpoint interval so the
reconstructed schedules can be checked against logged values.

For each interval, divide the integrated target linear-cooldown LR by the
integrated actual LR to obtain the retained update ratio. Tail differences of
these ratios give nonnegative checkpoint coefficients summing to one. Muon and
Adam must yield the same normalized interval ratios; incompatible schedules
stop this implementation. The weights approximate a cooldown with the observed
interval update directions fixed; they do not reproduce a retrained cooldown.

The merge streams native OCDBT arrays in FP32 with reads aligned to native storage
chunks. `AverageConfig.max_chunk_bytes` defaults to 512 MiB and must hold the
largest native FP32 chunk; a smaller value is rejected before weight reads. The
inventoried Hero layout has a largest chunk of 324 MiB. Reserve at least 8 GiB for
array buffers, decompression, write staging and Python/JAX/TensorStore runtime;
64 GiB of host RAM is recommended for the averaging worker. Recheck the largest
chunk and memory budget when changing the source layout. The output retains the
latest effective routing-control biases from Hero's quantile-balancing (QB)
router. The saved pending QB values set these per-layer/per-expert biases during
restore; they are retained from the latest source instead of averaged.
Optimizer and loader state are excluded.
`averaging.json` records coefficients, LR curves, norms, placement and source
provenance; completed `metadata.json` is published last.

Keep raw averaging candidate B separate. Evaluate it against latest PT candidate
A using predeclared loss, content and routing limits. If B fails or cannot be
produced, retain A as the selected base. Prepare candidate C with a one-source
`AverageConfig` and `initialize_donors=true`. This changes only the five specified
output columns; input embeddings are preserved. The pinned tokenizer must encode
each donor as the single token shown below. Leading spaces in donor text matter.

| Output token | Token ID | Donor text | Donor ID |
| --- | --- | --- | --- |
| `<\|start_header_id\|>` | 128006 | ` role` | 3560 |
| `<\|end_header_id\|>` | 128007 | ` message` | 1984 |
| `<\|start_think\|>` | 128002 | ` think` | 1781 |
| `<\|end_think\|>` | 128003 | ` answer` | 4320 |
| `<\|eot_id\|>` | 128009 | `<\|end_of_text\|>` | 128001 |

C's bound provenance records its A/B ancestry and exact forward configuration.
When selecting A, reuse its inventory `CheckpointPin`. When selecting B, create a
new pin with B's output path and digests of its completed `metadata.json` and
`manifest.json`. Reuse the original exact saved `config_path` and `config_digest`
from B's `averaging.json.request.checkpoints[0]`; `averaging.json` itself is not a
saved run configuration. Carry forward its tokenizer revision and durable
revision evidence. Use this single A/B pin in C's `AverageConfig.checkpoints`.

## Data and native evaluation

`format_data` accepts explicitly pinned canonical-message sources or can
materialize a bounded selection from existing instruction catalogs:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_data \
  catalog-selection.json artifacts/hero-format-sft/sources --materialize-catalog
```

The first command writes canonical JSONL files and a `sources.json` array of
pinned `SourceSpec` records. Copy that array into `DataRecipe.sources`; it is not
a recipe by itself. For example, after creating `data-recipe.json` with the
reviewed tokenizer and split settings:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

from experiments.grug.moe_hero_ep.format_data import DataRecipe

recipe_path = Path("data-recipe.json")
recipe = json.loads(recipe_path.read_text())
recipe["sources"] = json.loads(Path("artifacts/hero-format-sft/sources/sources.json").read_text())
validated = DataRecipe.model_validate(recipe)
recipe_path.write_text(validated.model_dump_json(indent=2) + "\n")
PY
uv run python -m experiments.grug.moe_hero_ep.format_data \
  data-recipe.json artifacts/hero-format-sft/data
```

The remaining `DataRecipe` fields are `tokenizer_path`, immutable
`tokenizer_revision`, optional `tokenizer_snapshot_digest`, `heldout_fraction`,
`split_seed` and `max_tokens` (at most 16,384).

Preparation uses the shared Marin template and assistant mask. It deduplicates
across sources, splits by conversation content, excludes whole overlength
conversations, and writes source weights, exact counts, token totals and hashes.
Train JSONL is separated by source so mixture weights remain effective.

Pin a `format_eval.Panel` for each enabled interface and serialization control.
The required initial coverage is simple chat (including system and multi-turn
examples) and an actual 16K retrieval check. Enable separate reasoning-on,
reasoning-off, tool-call and tool-observation panels when those interfaces are
part of the intended use. E adds an actual 65K panel. The two serialization
controls are the shared Marin template and explicitly pinned plain-text legacy
continuations; a legacy prompt is not a claim of universal PT formatting.
The native runner uses dropless routing, BF16-rounded weights, raw special tokens
and Marin EOT stopping. Legacy controls use the explicitly recorded prompts and
tokenizer EOS. It validates actual prompt lengths before weight I/O:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_sample \
  native-panel-request.json <fresh-result-directory>
```

For multiple panels on the same weights, prefer
[NativeSuiteRequest](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_suite.py). It checks
all inputs before restoring once, then retains each panel's raw samples, grading
report and optional content scores alongside PT loss/routing:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_suite \
  native-suite-request.json <fresh-suite-directory>
```

Set the required `pt_repetitions` to `3` for baseline A and `1` for B/C/D/E
(supported range: 1–10). The suite measures and saves PT loss/routing before
generation; its report retains every repeat and the observed loss range.
The dropless local evaluator requires `expert_axis_size: 1`; on the 64-GPU
rack use `replica_axis_size: 1`, leaving the mesh data axis at 64. This is
the existing sampler’s FSDP layout. The SFT training mesh is configured separately.

Use purpose `content` only for fixed-reference accuracy panels. Human freeform
formatting examples use purpose `format`. The PT/content panel identities stay
fixed across candidate comparisons; a later D formatting acceptance panel can
contain more cases than the initial formatting diagnostic.

`format_eval` grades completed samples, compares candidate evaluations, summarizes
smokes and calculates measured pilot budgets:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output format-report.json grade panel.json samples.json
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output comparison.json compare A.json parent.json child.json policy.json
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output pilot-budget.json pilot measured-smoke.json budget-inputs.json
```

Formatting reports retain every generation and failure and report exact counts
and Wilson intervals. A 99% observed fixed-panel rate is an operational gate;
it does not establish a population rate above 99%. Content, PT held-out loss,
routing and native-to-serving parity are separate inputs to candidate acceptance.
The `compare` command consumes assembled `CandidateEvaluation` records. Use
[CandidateHandoffRequest](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_handoff.py) to
bind the completed suite, its manifest digest, actual checkpoint ancestry,
format/content panel IDs and exact local PT panel bytes:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_handoff \
  candidate-handoff-request.json <fresh-candidate-directory>
```

The output `candidate.json` combines measured loss, routing, content outcomes and
retained formatting reports; `assembly.json` retains the checked provenance.
Use `candidate.json` as the comparison input. The assembler validates completed
suite output hashes and checkpoint ancestry. It leaves serving parity pending;
no manual boolean or filename establishes parity. Missing measurements leave
acceptance incomplete.

Use [format_measure](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_measure.py) to freeze
an explicitly selected bounded panel from existing PT validation caches, then
collect candidate loss and routing on those exact token/weight/segment bytes:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_measure \
  materialize pt-materialize-request.json pt-panel.json
uv run python -m experiments.grug.moe_hero_ep.format_measure \
  measure native-pt-request.json <fresh-pt-result-directory>
```

The source schemas `PtMaterializeRequest` and `NativePtRequest` pin the saved data
configuration, component, tag, row indices, tokenizer and checkpoint. Collection
uses no optimizer and keeps the routing-control biases in FP32. Repeated baseline
panels establish variability; choose regression limits before comparing derived
candidates. The format acceptance panel needs at least 500 simple-chat cases.
The fixed acquisition panel uses separate training-source targets and measures
learning during the smoke; it never supplies held-out acceptance results.

## Adapter training

`format_pipeline.FormatSftRecipe` binds the prepared dataset and exact C base to
`SFTSpec` and `sft_launch.HeroSftModel`. It keeps the current Hero backend and
packing/conversation isolation. The default command materializes a launch config:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_pipeline \
  recipe.json --output-prefix <durable-artifact-root> \
  --launch-config reviewed-launch.json
```

`FormatSftRecipe.gate_record` points to a `LaunchEvidence` JSON record containing
`baseline` (A), `parent` (selected A/B), `prepared` (C), `policy` and
`local_validation_reports`. The command recomputes C's regression decision and
checks the exact base and forward configuration. For the initial bounded learning
smoke, omit both optional `smoke` and `pilot_budget` fields. Declare that smoke's
own token/update/LR bounds in the recipe, together with the fixed acquisition
panel and baseline-derived loss/routing maximums.

After user launch authorization, add `--run` to submit training. Authorization is
a workflow prerequisite; `LaunchEvidence` does not store or enforce it. Set the
runtime `MARIN_PREFIX` to the reviewed `--output-prefix`. Each invocation creates
the `--launch-config` file exclusively, so use a fresh filename for submission.

The smoke writes raw update telemetry, checkpoint timings and
`telemetry/compiled-gradient-report.json` under its training output. Copy the
telemetry directory locally and assemble its measured record using the actual
warmup, microbatch, accumulation and optimizer settings:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output measured-smoke.json assemble-smoke <local-telemetry-directory> \
  --warmup-updates <warmup-updates> --microbatch-size <microbatch-size> \
  --gradient-accumulation <gradient-accumulation> --optimizer <optimizer>
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output smoke-report.json smoke measured-smoke.json
```

Review learning and compilation evidence before selecting a pilot budget. Pin
[SmokeRecord and PilotBudget](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_eval.py)
in the next launch's `LaunchEvidence.smoke` and `LaunchEvidence.pilot_budget`.
The launcher requires both together, recomputes the budget, and checks the
recipe's token cap, update count, warmup, learning rates and optimizer lineage.
It also requires the same model, adapter coverage/rank, data and acquisition
panels, global batch, mesh and resources as the measured smoke. When reusing
optimizer state, set `resume_checkpoint` to the measured terminal compact
checkpoint and `resume_sft_config` to its original exact `SftConfig`. Only the
four reviewed schedule fields (duration, warmup and the two learning rates) may
change; retained updates and token counts remain in the pilot budget.
Keep a full-SFT throughput control optional and report a speedup only when
weights, hardware and batch configurations match.

The adapter path freezes the BF16 base before differentiation. Adam state covers
only attention/shared-dense LoRA parameters and the five selected input/output
vocabulary deltas. Effective routing biases stay frozen. Compilation checks and
a learning smoke must establish that frozen weight-gradient and full-model
optimizer work are absent on the actual backend.

Compact checkpoints bind their base metadata, tokenizer and adapter configuration.
At successful completion, the backend automatically exports D to
`<training-output>/export/step-<completed-update>` and persists
`<training-output>/handoff.json`. No separate merge command is required. The
[HeroSftCheckpoint and HeroSftHandoff](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/sft_launch.py)
artifact exposes `native_checkpoint`, `metadata_digest` and `compact_checkpoint`;
the handoff JSON names the native digest `native_metadata_digest` and also records
the completed update, token counters and C base.

The export merges in FP32 onto the BF16-rounded C base used by training and
materializes ordinary Hero weights. Compare its logits with unmerged execution
at matching precision. Report an original-FP32-base merge separately if produced.
A completed export does not establish D's quality or serving-parity gates.

Use [MergeParityRequest](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/sft_merge_parity.py)
to measure compact-state versus ordinary-D parity on pinned real token rows:

```bash
uv run python -m experiments.grug.moe_hero_ep.sft_merge_parity \
  merge-parity-request.json <fresh-parity-directory>
```

The request pins `reference` (C), `sft`, C's exact source `model`, compact and
ordinary checkpoint metadata, a `PtPanel` path/raw SHA256, at most 32 explicit
`row_index`/`prediction_position` probes, and a compatible expert/replica mesh.
Declare maximum absolute logit error, probability error and greedy disagreement
count before measurement. Use the source forward backend and its compatible
mesh; the separate dropless evaluation configuration is not interchangeable
with C's recorded training configuration.

The command writes `report.json` with full-vocabulary errors, both greedy
choices, top scores and source/precision provenance, and exits nonzero if the
declared gates fail. It derives nonzero trained input rows from actual compact
state and requires each selected ID in a valid probe prefix in the same causal
segment. Include real rendered chat rows as well as PT rows: a plain C4 panel
can omit every chat input token. Every output column is compared regardless of
which IDs appear in the input. The local tiny roundtrip is an implementation
check; full-model merge parity remains an accelerator measurement and does not
establish serving parity.

The [serving parity helpers](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/testing/inference/backend_parity.py)
compare frozen native next-token scores with serving observations and check
same-rank repeatability. Use the candidate's exported HF weights and declared
probability tolerance on the same raw token IDs and prediction positions. The
existing Snowball runner is not a Hero or YaRN parity certificate. The configured
vLLM fork at `01911be34fac8715347962a8d791ca34a879e01e` ignores Hero YaRN settings
and computes router projections at different precision from the native BF16
path. A serving-side fix and actual GPU parity are required before claiming E
works in serving.

## 65K YaRN candidate

After format candidate D passes its gates and is merged, prepare candidate E:

```bash
uv run python -m experiments.grug.moe_hero_ep.context_extension \
  context-extension.json candidate-E.json
```

Populate `ContextExtensionConfig.source_checkpoint` from the handoff's
`native_checkpoint` and `metadata_digest` from `native_metadata_digest`. Set
`model` to D's original 16K forward configuration and `source_revision` to the
immutable implementation commit. Do not pass the compact adapter checkpoint.
The context-extension entrypoint checks the permanent D export and C ancestry;
record D's acceptance evaluations separately before proceeding.

E references D's arrays and changes forward configuration only: 16,384 to 65,536
positions, YaRN factor 4. `ContextExtensionConfig.beta_fast`, `beta_slow` and
`mscale` default to 32, 1 and 1. The Q/K rotary amplitude is
`1 + 0.1 * mscale * log(4)`; these values are persisted in E's `model.yarn` and
`hf_config` records. Hero's rotated half uses its original interleaved
pairing. Unrotated coordinates, rope-free global layers, sliding windows and QK
multipliers retain their source behavior.

The manifest marks E as pending validation. Compare D and E on identical short
and actual 16K panels, then evaluate E on actual prompts of at least 65,000 tokens.
Exported HF configuration carries the YaRN settings. Serving must implement the
same partial/interleaved scaling and pass parity before E is used downstream.
A configured 65K limit alone does not establish long-context competence.
