# PDBThink evaluation

Marin's `pdbthink` evaluation runs the published PDBThink coordinate panel through
TaskCompendium. It tests reasoning over supplied protein coordinates: distances,
contacts, residue environments, interfaces and changes between structures.
Answers are checked by the upstream parser and scorer without tools or an LLM judge.

This adds structured 3D geometry to the ID evaluation suite. Biomedical question
sets such as PubMedQA and MedQA mainly test knowledge and text interpretation;
GenomeQA tests sequence tasks. PDBThink asks the model to derive answers from
atom records supplied in the prompt. Its incremental value is this input and
reasoning format. Correlation with downstream research performance has not been
established.

Use it as an in-distribution development evaluation under the
[Marin evaluation policy](https://github.com/marin-community/marin/issues/10197).
Upstream publishes the panel's prompts, answers and model traces, so it is not a
contamination-resistant holdout. Exclude these questions and teacher completions
from SFT data and do not use the questions directly as RL tasks. The separate
[100k coordinate-task release](https://github.com/Open-Athena/pdbthink/blob/1bf06a464b2eab2f1e16c4dd16714314f225e196/docs/coordinate-task-generation.md)
is outside this integration.

## Pinned panel and scientific limits

The source and scorer use upstream commit
[`1bf06a4`](https://github.com/Open-Athena/pdbthink/tree/1bf06a464b2eab2f1e16c4dd16714314f225e196),
the latest upstream main revision checked on November 23, 2026. The panel is
`coordinate-report-v5.0.0`: 88 questions across 19 families, each with one
minimal-PDB rendering. It uses prompt v6 and scorer 1.2.0. Later report versions
add model conditions while retaining this scientific panel.

[Upstream revision 1.1](https://github.com/Open-Athena/pdbthink/blob/1bf06a464b2eab2f1e16c4dd16714314f225e196/docs/original-benchmark-v1.1.md)
corrected residue mappings, interface wording, clash-topology eligibility and
accepted contact sets. Subsequent changes preserve frozen system instructions
and reject exports that merely relabel older prompts. All 88 imported prompts
and answer keys match the
[published artifacts](https://github.com/Open-Athena/pdbthink/tree/1bf06a464b2eab2f1e16c4dd16714314f225e196/docs/benchmark-report/questions)
exactly.

This panel replaces the earlier Marin candidate bundle of 121 questions and
253 renders. It contains no mechanism questions, normalized-coordinate variants
or context-only controls, and excludes the defective 7WZ5 glycosylation-clash
question identified in the earlier audit. Historical GLM scores on that bundle
are not scores for this panel. The source/scorer code still shares the earlier
scientific revision; a new commit alone does not establish that every scientific
issue has been fixed.

The S05 prompts still omit quantitative fold-classification thresholds. The
upstream instance metadata marks all 88 questions `proposed`, and the panel has
not received an independent human scientific review. Interpret results with
these limits. The earlier audit is archived at
`gs://marin-us-west1/evaluation/pdbthink/full-quality-audit-20261112`
([Filer](https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-west1%2Fevaluation%2Fpdbthink%2Ffull-quality-audit-20261112)).

The committed manifest records the source revision, scientific revision,
dataset fingerprint, family counts and bundle SHA-256. Prompts and answer keys
are kept in the evaluation data cache, outside Marin's Git repository.

## Run

```bash
uv run python -m experiments.evaluation.cli launch \
  --model <registered-model-key> --evals pdbthink --dry-run
```

Remove `--dry-run` to submit. See [Running evaluations](../tutorials/run-lm-evals.md)
for resource selection, model configuration and result inspection. Use a model
configuration with its full supported context and output allowance; the catalog
may use a smaller context for other evaluations. The complete panel must fit
before inference starts. Oversized prompts fail preparation without cropping
or silently reducing the evaluated population.

Each question receives three completions at temperature 0.7, top-p 1 and disabled
top-k. The system and user messages are preserved exactly. Each request uses the
remaining model context after the full prompt and a 64-token reserve, unless an
explicit run or model output limit is supplied. Output limits, tokenizer revision,
request options, responses and grades are retained in the native archive.
A `--limit` run is a smoke test, not a full-panel score.

Long reasoning traces may exceed the launcher's default 30-minute request
deadline. Set `evaluation_request_timeout` in the model YAML to a sufficient
number of seconds; it extends the evaluator and inference-transport deadlines
together. This does not change the token allowance or launch identity. The
archive records each attempt's deadline; increasing it can resume an incomplete
run while preserving completed responses. Token budgets and scoring settings
must still match. Report timed-out attempts as incomplete coverage and include
their GPU time in the cost accounting.

Requests specify `tools: null` and `tool_choice: none`. No tool is executed.
Tool calls, refusals, malformed answers and output-limit finishes score zero.
Only final answer content is graded; an unfinished reasoning trace cannot supply
an answer. Transport failures remain unscored and make coverage incomplete.

The headline `macro_accuracy` matches the upstream report: compute fully correct
answer frequency within each family, then weight the 19 families equally.
`micro_accuracy` weights questions equally. The auxiliary `score` retains the
upstream partial-credit score, including set overlap. Three completions are
averaged without pass@k selection. Upstream report conditions use one completion
per question, so fresh Marin baselines have a different sampling count.

## Data and reproduction

The 1.4 MB immutable bundle is stored at
`gs://marin-us-west1/evaluation/datasets/Open-Athena/pdbthink/1bf06a464b2eab2f1e16c4dd16714314f225e196/evaluation.json.gz`
([Filer](https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-west1%2Fevaluation%2Fdatasets%2FOpen-Athena%2Fpdbthink%2F1bf06a464b2eab2f1e16c4dd16714314f225e196%2Fevaluation.json.gz)).
The catalog reads `mirror://evaluation/datasets/Open-Athena/pdbthink/<revision>/evaluation.json.gz`,
which resolves beneath the worker's Marin storage prefix. GCS mirrors can copy
this small object from the canonical cache. CoreWeave workers use their regional
S3 copy; stage it before launching in a new CoreWeave region:

```bash
uv run fsutil cp \
  gs://marin-us-west1/evaluation/datasets/Open-Athena/pdbthink/1bf06a464b2eab2f1e16c4dd16714314f225e196/evaluation.json.gz \
  s3://marin-us-east-02a/marin/evaluation/datasets/Open-Athena/pdbthink/1bf06a464b2eab2f1e16c4dd16714314f225e196/evaluation.json.gz
```

Acquisition verifies the checksum and stages the source with the evaluation
corpus. Workers need access to their regional private cache. Missing or changed
bytes fail before inference.

The frozen source dataset is archived beside the bundle as
`source-dataset.tar.gz`
([Filer](https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-west1%2Fevaluation%2Fdatasets%2FOpen-Athena%2Fpdbthink%2F1bf06a464b2eab2f1e16c4dd16714314f225e196%2Fsource-dataset.tar.gz)).
Unpack it, then rebuild and verify the bundle with:

```bash
uv run --package marin-taskcompendium --extra pdbthink \
  python -m taskcompendium.importers.pdbthink_prepare \
  --dataset /path/to/report-coordinate-v5 \
  --cache-prefix /path/to/evaluation-cache
```

This validates the upstream dataset, checks its fingerprint, and requires the
complete generated manifest to match before writing. Rebuilding
`configs/dataset_v1.yaml` produces a different candidate cohort and must not be
substituted for the published report panel.

## Cost accounting

Scoring is deterministic CPU work; LLM judge cost is zero. Inference cost depends
on the model, serving topology and reasoning length. Report GPU-hours as the
allocated GPU count multiplied by serving wall time, including startup and idle
time. Retain token totals and full-panel coverage alongside accuracy. A dollar
estimate requires the applicable GPU-hour rate; missing billing data is not a
zero-cost run.
