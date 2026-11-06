# PDBThink evaluation

Marin's `pdbthink` evaluation runs the original protein structural-reasoning
benchmark through vendored Evalchemy. It is an in-distribution development eval
under the [Marin evaluation policy](https://github.com/marin-community/marin/issues/10197).
Exclude its questions and teacher completions from SFT data, and do not use its
questions directly as RL training tasks. The separate 100k coordinate-task
training dataset is outside this integration.

Questions ask models to reason from supplied protein coordinates, for example
to count residues, measure an atom-to-atom distance, or identify contacting
residue pairs. A semantic question is one underlying problem; a render is that
problem expressed using a particular coordinate format or control condition.

## Frozen benchmark

The source and scorer are pinned to
[`94b65a9`](https://github.com/Open-Athena/pdbthink/tree/94b65a918140e44ca6b6e94032a0ee0a7441b1dd).
Building that revision's `configs/dataset_v1.yaml` produces 121 semantic
questions and 253 renders: 158 minimal-PDB, 47 normalized-coordinate, and 48
context-only prompts. All questions have upstream curation status `proposed`.
This is a candidate benchmark, not an accepted final release.

Minimal-PDB renders use the Protein Data Bank text format. Normalized-coordinate
renders express the coordinates in tables; context-only controls omit them.
The 158 minimal-PDB renders include 31 rotated controls, which change the
coordinate frame while preserving the answer.

The frozen build uses prompt version `v5` and scorer version `1.1.0`. The older
candidate manifests committed upstream identify prompt version `v2`; scores
against those prompts are not directly comparable. Marin records the source,
config, definitions, structure and bundle hashes in its
[manifest](https://github.com/marin-community/marin/blob/main/vendor/evalchemy/eval/chat_benchmarks/PDBThink/manifest.json).
Prompts and gold answers live in the evaluation data cache and are not committed
to Marin.

## Prepare the evaluation cache

Build from a clean checkout of the pinned upstream revision. Run these commands
inside that checkout; acquisition needs network access to the upstream structure
sources, and subsequent build and validation use the cache:

```bash
git checkout --detach 94b65a918140e44ca6b6e94032a0ee0a7441b1dd
uv run --python 3.12 --extra tokenizer structural-reasoning acquire --config configs/dataset_v1.yaml
uv run --python 3.12 --extra tokenizer structural-reasoning build \
  --config configs/dataset_v1.yaml --output datasets/marin-v1
uv run --python 3.12 --extra tokenizer structural-reasoning validate --dataset datasets/marin-v1
```

From the Marin repository, validate and stage the bundle:

```bash
uv run --project vendor/evalchemy --frozen --extra pdbthink \
  python vendor/evalchemy/scripts/benchmarks/prepare_pdbthink.py \
  --dataset /path/to/pdbthink/datasets/marin-v1 \
  --cache-prefix /path/to/eval-cache
```

The command rejects any build that differs from the frozen manifest and verifies
the written bundle. It creates
`Open-Athena/pdbthink/94b65a918140e44ca6b6e94032a0ee0a7441b1dd/evaluation.json.gz`
under the cache prefix. The bundle is about 3.7 MB. For a distributed evaluation,
use a private object-store prefix accessible to the evaluation worker in its
region. Use `uv run fsutil cp` to copy an already prepared bundle to that prefix.

Set `dataset_cache` in a copy of
`experiments/evaluation/configs/evalchemy/pdbthink.yaml` to this cache prefix, then
launch the copied configuration:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model <registered-model-key> \
  --evalchemy-config /path/to/pdbthink.yaml \
  --dry-run
```

Remove `--dry-run` to submit. The registered `--evals pdbthink` configuration
uses the launcher's shared regional Evalchemy cache; prepare the bundle there
before using that shortcut. A missing or changed bundle fails preparation before
generation. See [Running evaluations](../tutorials/run-lm-evals.md) for model
configuration and launch options.

## Generation and scoring

Each render receives three sampled completions at temperature 0.7, preserving
the upstream system and user messages. Requests set `tools: []` and
`tool_choice: none`. An unexpected tool call scores zero. No tool execution or
LLM judge is involved.

Before the first request, Evalchemy counts every selected prompt with the served
model's tokenizer and chat template. An oversized prompt fails the task; it is
never cropped or silently skipped. Each output budget uses the remaining context
after a 64-token safety margin, bounded by the YAML `max_tokens` cap when supplied.
The saved results include per-render token budgets. A capped `--limit` run is a
smoke test; use all 253 renders for a full benchmark score.

The pinned upstream parser and scorer interpret `FINAL` answers, including
numeric tolerances and partial credit for sets. Malformed answers, refusals and
truncations score zero. Transport failures remain unscored infrastructure
failures and make the run incomplete. Successful requests are reused on resume.

The headline `macro_score` averages scores within each question family (a group
of questions testing the same operation), then gives each family equal weight.
It includes base minimal-PDB and normalized-coordinate
renders and excludes context-only and rotated controls. Completions are averaged;
there is no pass@k selection. The upstream report supplies protein-cluster
bootstrap intervals, family scores, control comparisons, consistency, failure
counts and mechanistic diagnostics under `pdbthink_report`. The frozen manifest
and actual token budgets are saved under `pdbthink` in the task results.
