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

## Completed exploratory experiment

[Issue #10428](https://github.com/marin-community/marin/issues/10428) records the
completed 2026-10-22 campaign and its failures. The LR-aware average of source
steps 282000, 288000 and 294000 used coefficients 0.232123642, 0.499105160 and
0.268771198. On the identical fixed 59-sequence C4 panel, aggregate next-token
cross-entropy was 2.130703034 for A (step 294000) and 2.109083774 for B (the
three-checkpoint average), a relative reduction of 1.01465%. The panel raw SHA256
was `e1c77e834b636f2889595751b10b3061c0d4583824aa5144808cba7a0bd4b880`. This does not reproduce the optimizer trajectory or establish the
quality benefits of a retrained cooldown.

After donor initialization, compact LoRA plus vocabulary training completed
64 total updates. The first eight updates span recovery segments; the longer
segment restored full compact state at update 8 and performed 56 additional
updates. Its linear schedule horizon extended from 8 to 64 while preserving
Adam, schedule counts, seed and deterministic data position. The first resumed
learning rate therefore rose relative to the earlier short cooldown tail.
A fresh 64-update run has a different learning-rate history.

| Setting | Executed value |
| --- | --- |
| Source context | 16,384; no YaRN |
| Training batch and mesh | 64 sequences; expert64/context1/replica1 |
| LoRA | Rank 16, alpha 16; attention and shared dense projections |
| Vocabulary deltas | Five selected input rows and output columns |
| Adam peak learning rates | Adapters 1e-4; vocabulary 1e-5 |
| Warmup and seed | 2 updates; 20261021 |
| Cumulative processed-token cap | 8,388,608 for smoke; 67,108,864 after extension |
| Compact checkpoint cadence | Every update, including full Adam and counters |
| Numerical abort bounds | Loss 20, gradient norm 1000, router z-loss 100 |

Those abort bounds are numerical limits, not quality criteria. The final
cumulative counts were 65,975,980 processed and 42,664,754 supervised tokens.
On a fixed training-source teacher-forced EOT diagnostic, mean NLL changed from
3.3814 at step 8 to 0.4288 at step 64 and top-1 matches from 1/16 to 14/16.
These are EOT-token diagnostics, not average sequence NLL or held-out quality.

The separate 100-response qualitative panel contains 40 math tasks, 40 Python
tasks and 20 fixed-history follow-ups. All responses ended with EOT; none hit
the 2,048-token cap. Median length was 113.5 tokens and maximum length 496.
Manual review found 33 correct final math answers, four wrong, two incomplete
and one ambiguous reference. One correct final answer contradicted an earlier
number. Of the code implementations, 32 appeared plausible by static inspection
and eight had functional defects; generated code was not executed. Eighteen
fixed-history replies clearly met their request, one lost track of possession,
and one had a borderline item-count issue depending on whether drawing supplies
count as a kit or separate physical items.

No literal role continuation or pathological repetition was observed. One math
reply contained a stray closing thinking marker, and one code reply emitted an
end/start-of-text pair before EOT. EOT termination therefore does not establish
clean serialization. These results support provisional capacity measurements on
the intended RL backend. Full held-out/native format/content evaluation,
independent merge and serving parity, and long-context behavior remain
unresolved. Native full-prefix sampler timings do not measure serving throughput.

## Reproduce the exploratory campaign

The completed campaign used a separately authorized exploratory path. It did
not complete the formal quality-qualified pilot workflow described later in this
page. `sft_campaign` retains optimizer/counter/data state and explicit numerical
bounds without fabricating that qualification. The original completed SFT and
rollout model implementation was commit
`4d97008b7e7626fe9eb517c196988741b48cd6a6`; construction used
`2416e9d797b0fbbb23327105618c56f823d42844` and the matched A/B evaluator used
`47c6a090cfb6c5d875079264fa25cbf3bf20d737`. The public campaign and rollout
entrypoints bind a new repetition to the clean commit containing these modules.
Record that full commit in each request and receipt. Do not label a repetition
at a different revision as a byte-for-byte replay of the original execution.

First follow the inventory, averaging, donor and data procedures below. The
executed data selection was the first 6,000 scanned training rows from the
existing `HuggingFaceH4/no_robots` catalog adapter at full Hub revision
`e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`, weight 1.0. Use tokenizer
`marin-community/marin-tokenizer` at revision
`a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2`, held-out fraction 0.2, split seed
20261019 and maximum length 16,384. Preparation produced 4,806 training and
1,186 held-out conversations and 1,385,205 non-padding training tokens. Keep the
source revision and emitted JSONL hashes; an `in_pretraining=false` declaration
does not establish absence from pretraining. Build the training-source
acquisition panel with `build_acquisition_panel(..., max_examples=16)` and the
five selected token IDs. Retain its digest separately from held-out panels.

Write a `BaseReference` JSON record from C's actual checkpoint metadata and
ancestry, then construct the initial request using the public dataclass. The
following example reads caller-supplied small metadata files; it does not read
model arrays. `prepared/` is the output of `format_data` above.

```python
import json
import subprocess
from pathlib import Path

import draccus
from fray.types import GpuConfig, ResourceConfig

from experiments.grug.moe_hero_ep.averaging import read_prepared_provenance
from experiments.grug.moe_hero_ep.format_eval import AcquisitionPanel
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.sft import BaseReference, SftConfig
from experiments.grug.moe_hero_ep.sft_campaign import SftCampaign

reference = draccus.decode(BaseReference, json.loads(Path("base-reference.json").read_text()))
prepared = read_prepared_provenance(reference.checkpoint, reference.metadata_digest)
manifest = json.loads(Path("prepared/manifest.json").read_text())
acquisition = AcquisitionPanel.model_validate_json(Path("acquisition.json").read_text())
campaign = SftCampaign(
    name="hero-format-smoke-reproduction", version="2026.10.21",
    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    reference=reference, model=draccus.decode(GrugModelConfig, prepared["model"]),
    sft=SftConfig(16, 16.0, (128006, 128007, 128002, 128003, 128009),
                  1e-4, 1e-5, 8, 2, 20261021),
    data_root="prepared", data_manifest="prepared/manifest.json",
    data_manifest_digest=manifest["manifest_digest"],
    acquisition_panel="acquisition.json", acquisition_panel_digest=acquisition.panel_digest,
    cache_prefix="local-chat-cache", output_root="FRESH_REMOTE_SMOKE_ROOT",
    resources=ResourceConfig(cpu=120, ram="890g", disk="1t",
                             device=GpuConfig(variant="GB200", count=4), replicas=16),
    batch_size=64, expert_axis_size=64, context_axis_size=1, replica_axis_size=1,
    max_processed_tokens=8388608,
    maximum_metrics=(("loss", 20.0), ("gradient_norm", 1000.0),
                     ("train/router/router_z_loss", 100.0)),
)
Path("smoke-local-cache.json").write_text(json.dumps(draccus.encode(campaign), indent=2) + "\n")
```

Replace `FRESH_REMOTE_SMOKE_ROOT` with a fresh artifact destination. Publish the
prepared JSONL, manifest, acquisition panel and request metadata where every
worker can read them; update the request's paths to those locations. Build chat
caches on CPU under a local prefix, then publish the completed cache prefix with
`uv run fsutil verified-copy LOCAL_PREFIX REMOTE_PREFIX` and preserve the
verification manifest. Change `cache_prefix` to that published prefix before
reviewing the launch request. Keep `version`, data bytes and template unchanged
between segments so their cache fingerprint stays fixed. GPU workers require a
completed matching cache and never invoke tokenization.

```bash
uv run python -m experiments.grug.moe_hero_ep.sft_campaign \
  smoke-local-cache.json --build-caches
uv run python -m experiments.grug.moe_hero_ep.sft_campaign \
  smoke.json --review smoke-review.json
```

The reviewed request must point to the published metadata/data/cache paths.
To reproduce the extension, publish the exact eight-update request bytes at an
immutable location accessible to workers. After the smoke succeeds, construct
the next request from that same published request. `SMOKE_REQUEST_URI` and
`FRESH_REMOTE_LONG_ROOT` below are caller-supplied artifact locations.

```python
import dataclasses
import hashlib
import json
from pathlib import Path

import draccus

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft_campaign import (
    RecoveryParent, latest_completed_checkpoint, read_campaign,
)

smoke, raw = read_campaign("SMOKE_REQUEST_URI")
selected = latest_completed_checkpoint(smoke)
if selected is None or selected["step"] != 8:
    raise ValueError("The eight-update parent must be complete before extending its schedule")
parent = RecoveryParent(
    request_uri="SMOKE_REQUEST_URI",
    request_sha256=hashlib.sha256(raw).hexdigest(), request_digest=digest(json.loads(raw)),
    **{key: selected[key] for key in ("path", "step", "metadata_digest", "counters")},
)
long = dataclasses.replace(
    smoke, name="hero-format-long-reproduction", output_root="FRESH_REMOTE_LONG_ROOT",
    sft=dataclasses.replace(smoke.sft, num_steps=64),
    max_processed_tokens=67108864, recovery_parent=parent,
)
Path("long.json").write_text(json.dumps(draccus.encode(long), indent=2) + "\n")
```

Review `long.json` with `sft_campaign --review` before submission. Preserve the
remaining fields, including `version`. Preparation rejects changed data, base,
peak learning rates, warmup, packing or optimizer settings. Own later completed
checkpoints take precedence on native preemption recovery; unfinished diagnostic
queues survive even when all gradient updates have finished.

Use the shared `format_job.LaunchConfig` for either worker. Its request SHA256
pins the exact worker bytes copied as `request.json`. Set the source revision to
the clean committed checkout, the resources above, four processes per task,
and the selected controller/placement. The SFT command is
`["python", "-m", "experiments.grug.moe_hero_ep.sft_campaign", "request.json", "--worker"]`.
Copy the model's `moe_implementation` and `remat_mode`, the request's output root
and published cache prefix into the launch record. Standard Iris mounted
credentials are inherited; never put credential values in requests or receipts.
A matching current Iris client/controller is required. For example, bind the
reviewed eight-update request to a direct GPU controller as follows; use
`long.json` for the later segment and a fresh receipt for every submission.

```python
import hashlib
import json
from pathlib import Path

import draccus

from experiments.grug.moe_hero_ep.format_job import LaunchConfig
from experiments.grug.moe_hero_ep.sft_campaign import read_campaign

campaign, raw = read_campaign("smoke.json")
launch = LaunchConfig(
    name=campaign.name, source_revision=campaign.source_revision,
    controller_cluster="GPU_CLUSTER", resources=campaign.resources,
    processes_per_task=4,
    command=("python", "-m", "experiments.grug.moe_hero_ep.sft_campaign", "request.json", "--worker"),
    request_path="smoke.json", request_sha256=hashlib.sha256(raw).hexdigest(),
    output_root=campaign.output_root, moe_implementation=campaign.model.moe_implementation,
    remat_mode=campaign.model.remat_mode, cache_prefix=campaign.cache_prefix,
)
Path("launch.json").write_text(json.dumps(draccus.encode(launch), indent=2) + "\n")
```

```bash
uv run python -m experiments.grug.moe_hero_ep.format_job launch.json
# After inspecting the dry-run and authorizing this specific run:
uv run python -m experiments.grug.moe_hero_ep.format_job launch.json \
  --submit --receipt fresh-receipt.json
```

The launcher defaults to interactive priority, six hours per task, 24 hours
scheduling, ten preemption retries and zero application-failure retries. It
refuses existing output/receipt/job identities. A pending receipt after a lost
submission response requires job inspection, not a second submission. Use the
[Iris monitoring workflow](https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md)
to verify all ranks, restore, finite telemetry and completed checkpoints.

At successful completion, the SFT worker automatically runs `export_sft`, merging
LoRA and vocabulary deltas into ordinary BF16 weights while preserving routing
control state. It writes both `adapters/step-64` (full compact Adam/state) and
`export/step-64` (ordinary native weights), then `handoff.json`. There is no
additional training or separate merge job. Resolve the successful
`completions/<leaderUID>.json` record and download that small JSON as
`downloaded-completion.json`. Build the native sampling inputs from its handoff:

```python
import json
from pathlib import Path

import draccus
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, digest
from experiments.grug.moe_hero_ep.sft_campaign import read_campaign

long, _ = read_campaign("long.json")
handoff = json.loads(Path("downloaded-completion.json").read_text())["handoff"]
metadata = json.loads((StoragePath(handoff["native_checkpoint"]) / "metadata.json").read_text())
if digest(metadata) != handoff["native_metadata_digest"] or handoff["step"] != 64:
    raise ValueError("Native export differs from the completed handoff")
checkpoint = Checkpoint(
    uri=handoff["native_checkpoint"], metadata_digest=handoff["native_metadata_digest"],
    step=64, run_id=long.name, timestamp=metadata["timestamp"],
)
Path("checkpoint.json").write_text(checkpoint.model_dump_json(indent=2) + "\n")
Path("model.json").write_text(json.dumps(draccus.encode(long.model), indent=2) + "\n")
```

Prepare the frozen panel, replacing `IMPLEMENTATION_COMMIT` with the same clean
commit used for the sampling launch:

```bash
uv run python -m experiments.grug.moe_hero_ep.rollout_sanity prepare \
  --checkpoint checkpoint.json --source-model model.json \
  --source-revision IMPLEMENTATION_COMMIT --target-cluster GPU_CLUSTER \
  --destination FRESH_REMOTE_ROLLOUT_ROOT \
  --train-jsonl prepared/train.jsonl --heldout-jsonl prepared/heldout.jsonl \
  --output fresh-rollout-preparation
```

Use a separate `LaunchConfig` with the prepared rollout request and worker command
`["python", "-m", "experiments.grug.moe_hero_ep.rollout_sanity", "run", "request.json"]`.
Its model/runtime fields must describe the explicit dropless evaluation model,
not the capacity-limited training model. The worker verifies 64 GPU processes,
restores once, and saves ordered 64/36 waves with the original per-case seeds,
temperature 0.7 and 2,048-token cap. The source architecture stays 16K; physical
padding is 4,096 because every prompt plus its full cap fits. Generation uses
full-prefix recomputation without a KV cache. It does not establish serving
throughput, capacity-limited expert64 behavior or 16K runtime coverage.

Resolve `completions/*.json` to the actual `attempts/<leaderUID>` directory.
Verify its manifest, request, checkpoint, panel and raw file hashes. Prefer the
hash-bound `transcripts.json` derived view for local review; only probability
payloads are omitted, while original samples remain remote. Inspect every
response against the frozen references and their ambiguity notes. Treat cap
hits as unfinished/censored separately from role leakage, repetition or content
errors. The references support manual review; no generated code is executed.

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
suite output hashes and checkpoint ancestry. Serving parity stays pending unless
the request supplies both `serving_parity_report` and its raw
`serving_parity_sha256`. Supply the report written by `collect`, with a hash of
that report's bytes; the `verify` output is a separate inspection artifact.
The assembler recomputes that report's verdict and
checks its candidate, checkpoint, forward model, tokenizer and suite identity.
Missing measurements leave acceptance incomplete.

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

The smoke writes raw update telemetry, checkpoint timings,
`telemetry/gradient-structure-report.json` and
`telemetry/compact-optimizer-report.json` under its training output. The optimizer
report checks the exact compact parameter and Adam moment trees, FP32 dtypes,
counter values and buffer sizes before the first update. The gradient report
records the actual compact gradient output paths, shapes and dtypes. It does not
inspect internal kernel allocations or require a general HLO absence certificate.
Independent numerical LoRA checks, finite updates and the declared diagnostic
limits remain required for the bounded learning smoke.

Acquisition evaluations run after the current invocation finishes its permitted
gradient updates. Each evaluation restores the adapter and vocabulary parameters
from that update's compact checkpoint. This path requires `save_every_steps=1`.
Core update telemetry is saved before checkpoint publication; a queue pins those
telemetry bytes by SHA256 and records unfinished acquisition measurements.
Checkpoint restore recovers the queue even when no gradient updates remain.
Final `update-N.json` reports are written after their acquisition measurements.
This avoids alternating the large training executable and the smaller acquisition
executable: overlapping NCCL symmetric-memory registrations can make the next
training collective fail on the ragged backend
([prior diagnosis](https://github.com/marin-community/marin/issues/8861#issuecomment-5531532861)).

Smoke qualification pins the gradient-structure and compact-optimizer reports'
raw SHA256 hashes and checks their execution, base and measurement identities.
It checks FP32 output shapes for paired LoRA factors and selected vocabulary
deltas, their total parameter bytes, fresh compact Adam state and measured
nonzero input-delta gradients. These checks establish the compact gradient
interface and activation backpropagation. Internal kernel allocations remain
outside this evidence; compiler audits are optional diagnostics.

Copy the telemetry directory locally and assemble its measured record using the actual
warmup, microbatch, accumulation and optimizer settings:

```bash
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output measured-smoke.json assemble-smoke <local-telemetry-directory> \
  --warmup-updates <warmup-updates> --microbatch-size <microbatch-size> \
  --gradient-accumulation <gradient-accumulation> --optimizer <optimizer>
uv run python -m experiments.grug.moe_hero_ep.format_eval \
  --output smoke-report.json smoke measured-smoke.json
```

Review learning, gradient and optimizer evidence before selecting a pilot budget. Pin
[SmokeRecord and PilotBudget](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_eval.py)
in the next launch's `LaunchEvidence.smoke` and `LaunchEvidence.pilot_budget`.
Qualification rechecks the retained gradient and optimizer reports before
accepting a budget. Missing runtime measurements leave the budget unqualified.
The launcher requires both together, recomputes the budget, and checks the
recipe's token cap, update count, warmup, learning rates and optimizer lineage.
Set `PilotBudget.format_panel_id` to the pinned D acceptance panel, including at
least 500 simple-chat cases. It can differ from the smaller initial C diagnostic.
It also requires the same model, adapter coverage/rank, data and acquisition
panels, global batch, mesh and resources as the measured smoke. When reusing
optimizer state, set `resume_checkpoint` to the measured terminal compact
checkpoint and `resume_sft_config` to its original exact `SftConfig`. Only the
four reviewed schedule fields (duration, warmup and the two learning rates) may
change; retained updates and token counts remain in the pilot budget.

Pilot training ends the current segment at the first unevaluated early-update
or actual non-padding-token inspection boundary. An inspection can overshoot
its token target by one complete update; the overall token cap remains strict.
A reused smoke that already reached a boundary pauses before further updates.
The segment saves compact state, exports ordinary weights and writes immutable
`inspection.json`. Its status distinguishes `awaiting_evaluation` from
`budget_exhausted`; a paused export does not establish acceptance.

Evaluate that exact ordinary export with the pinned native suite and assemble
its candidate handoff. To continue, set `LaunchEvidence.continuation` using the
[PilotContinuation](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/format_pipeline.py)
schema: the prior `inspection_path`, the canonical digest of its entire parsed
JSON as `inspection_digest`, and the exact `CandidateHandoffRequest` as
`candidate_handoff`. Use the inspection's compact checkpoint for
`resume_checkpoint` and keep `resume_sft_config` equal to the entire pilot's
unchanged `SftConfig`. Compute the inspection digest with
[the shared digest helper](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/ops/vibe_check/completions.py):
SHA256 of UTF-8 `json.dumps(record, sort_keys=True, separators=(",", ":"))`.
For each segment, choose a fresh `recipe.name` and unused `--output-prefix`, and
set runtime `MARIN_PREFIX` to that prefix. Compact state, ordinary export and
`inspection.json` are written beneath that segment's training artifact path.

Continuation recomputes the actual held-out candidate measurements and requires
passing loss, content and routing gates against both C and A. Only measured
formatting failures permit further training. Once native formatting and content
gates pass, training stops even if serving parity remains pending. Missing
measurements, regression, exhausted budgets or changed state/data/policy/budget
identities reject continuation. The whole-pilot LR schedule, Adam moments,
cumulative counters and data cursor persist across segments. Every further
training launch still requires authorization.

Keep a full-SFT throughput control optional and report a speedup only when
weights, hardware and batch configurations match.

The adapter path freezes the BF16 base before differentiation. Adam state covers
only attention/shared-dense LoRA parameters and the five selected input/output
vocabulary deltas. Effective routing biases stay frozen. Independent numerical
checks and the learning smoke validate the compact training path. General
compiler certificates are optional diagnostics, not an acceptance gate.

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

Use [ServingParityRequest](https://github.com/marin-community/marin/blob/main/experiments/grug/moe_hero_ep/serving_parity.py)
to select bounded greedy trajectories from a completed native suite and bind
them to the candidate's completed version-3 HF export. Pin the maximum absolute
probability error and serving source. Each selected data-parallel rank repeats
each trajectory in at least two waves. Declare the minimum number of compared
KV decode positions per trajectory, excluding its first prefill prediction.
Include short and actual 16K prompts plus an
EOT-stopping trajectory; E additionally requires an actual prompt of at least
65,000 tokens. Each prompt and completion must fit the model context.

```bash
uv run python -m experiments.grug.moe_hero_ep.serving_parity \
  prepare serving-parity-request.json <fresh-preparation.json>
uv run python -m experiments.grug.moe_hero_ep.serving_parity \
  collect serving-parity-request.json <fresh-serving-report.json>
uv run python -m experiments.grug.moe_hero_ep.serving_parity \
  verify serving-report-pin.json <fresh-verification.json>
```

The verify input is a `JsonPin` containing the report's `path` and raw `sha256`.
Preparation reads bounded metadata and retained native scores, without restoring
weights. Its output is an optional preflight inspection artifact; `collect`
repeats preparation from the pinned request and does not consume that output.
Collection requires an `OwnedServer` JSON record, using the schema in the same
module, supplied as `ServingParityRequest.launch_manifest` (`path` and raw
`sha256`). Construct it from the owned execution's actual launch records. It
binds the exact
ModelSpec, export, released wheel/source/hash, passed GPU qualification and
actual verified-wheel/topology startup records. Missing runtime evidence leaves
the report pending and makes no HTTP requests. This workflow does not launch a
server, publish a wheel or change the serving pin.

Collection retains raw token requests and multi-token responses. Verification
recomputes top-token probability errors on identical native prefixes, KV decode
coverage, EOT stopping and exact same-rank repeatability. Divergence that loses
required decode coverage cannot pass. Prefix caching and speculative decoding
must be disabled. This report consumes physical runtime qualification; it does
not establish kernel or layer parity itself.

The configured vLLM fork at `01911be34fac8715347962a8d791ca34a879e01e`
omits Hero YaRN wiring and uses different router projection precision from the
native BF16 path. A qualified serving-side fix and actual GPU observations are
required before E can pass this gate.

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
