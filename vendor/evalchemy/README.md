# Evalchemy

Marin runs this vendored evaluator in a separate Python 3.12 environment against
OpenAI-compatible model endpoints. Its `uv.lock` resolves evaluation dependencies
independently from the Marin workspace. FineStore and rigging use the local Marin
sources so evaluator changes and their storage dependencies can land together.

The source comes from
[`marin-community/evalchemy`](https://github.com/marin-community/evalchemy/tree/09d27c93a8846531e84fb30cfa11dacd2ef404c5)
at commit `09d27c93a8846531e84fb30cfa11dacd2ef404c5`. It retains all 42 custom
benchmark sources, their data and preparation scripts, lm-eval task overrides,
graders, streaming execution, FineStore output and resume support. Marin owns
serving and evaluation orchestration. Database tracking, Curator dispatch,
W&B reporting, standalone serving launchers, benchmark websites and unused
leaderboard outputs have been removed.

The evaluator's import packages, `eval` and `evalchemy_config`, belong to this
environment. Marin calls the `evalchemy` CLI in a child process.
Configuration validation requires the installed lm-eval task catalog; missing
catalog dependencies fail validation instead of using an incomplete task list.

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
