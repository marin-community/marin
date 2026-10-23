# Evalchemy

Marin runs this vendored evaluator in a separate Python 3.12 environment against
OpenAI-compatible model endpoints. Its `uv.lock` resolves evaluation dependencies
independently from the Marin workspace. FineStore and rigging use the local Marin
sources so evaluator changes and their storage dependencies can land together.

The source comes from
[`marin-community/evalchemy`](https://github.com/marin-community/evalchemy/tree/09d27c93a8846531e84fb30cfa11dacd2ef404c5)
at commit `09d27c93a8846531e84fb30cfa11dacd2ef404c5`. It retains all 42 custom
benchmark dataset sources, their required data and preparation scripts, lm-eval task overrides,
graders, streaming execution, FineStore output and resume support. Marin owns
serving and evaluation orchestration. A local GraphWalks adapter adds the pinned
`openai/graphwalks` dataset to the shared execution and grading flow, selecting
examples with the served tokenizer and preserving their output budgets.
Database tracking, Curator dispatch,
W&B reporting, standalone serving launchers, benchmark websites and unused
leaderboard outputs have been removed. Nested benchmark packages retain only
the dataset, prompt-processing and scoring code used by this runtime. Their
original model providers, rollout drivers and unused evaluator configurations
are removed.

The evaluator's import packages, `eval` and `evalchemy_config`, belong to this
environment. Marin calls the `evalchemy` CLI in a child process.
Configuration validation requires the installed lm-eval task catalog; missing
catalog dependencies fail validation instead of using an incomplete task list.

Large canonical datasets are fetched from immutable Hugging Face revisions.
`eval.data_sources.fetch_hf` returns a cached file path or object-store URL.
`fetch_url` caches direct HTTP(S) files by URL and an explicit immutable publisher
version under a selected prefix. Both helpers reuse completed staged files without
contacting the origin; Rigging's `open_url` opens their returned paths and does not itself
cache HTTP URLs. Marin uses its configured scratch store; standalone HF reads
use the native local HF cache, checking pinned snapshots before downloading.
`--dataset_cache` selects an explicit S3 prefix or local directory; HTTP callers
must pass a cache prefix to `fetch_url`. Small
datasets and custom snapshots remain bundled. See the
[dataset caching reference](../../docs/explanations/evaluation.md#dataset-caching)
for both APIs.

```bash
uv run --project vendor/evalchemy --frozen --no-dev evalchemy --help
uv run --project vendor/evalchemy --frozen --no-dev evalchemy validate-config experiments/evaluation/configs/evalchemy/gsm8k-smoke.yaml
```

Benchmark extras retain the fork's names. Select the extras needed by an
evaluation, such as `--extra humanevalplus --extra mbppplus --extra ifbench`.
Run retained local tests from this directory so source-relative fixtures and
child processes use the same evaluator:

```bash
cd vendor/evalchemy
uv run --frozen --group test --all-extras pytest
```

`eval/sources/__init__.py` lists the custom sources. `eval/lm_eval_tasks/` contains
the lm-eval overrides. Preserve their scoring and result contracts when changing
dependencies. The retained tests exercise request settings, grading, sample
identity, failure handling and resume behavior without launching model servers.

Nested benchmark licenses and citations remain with their source. The imported
fork has no root license file; this vendor does not assign a new license to that
source. `CITATION.cff` preserves Evalchemy's attribution.
