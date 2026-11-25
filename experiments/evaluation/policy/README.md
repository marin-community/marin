# Eval policies

An eval policy says which models (candidates) run which benchmarks, with which settings, and where
a candidate departs from them. Each version is a Python module, `versions/<version>/policy.py`, beside a
README that lists its candidates, its evaluations and how it differs from earlier policies. Marin's
policy is tracked in [#10197](https://github.com/marin-community/marin/issues/10197).

| File | Holds |
| --- | --- |
| [`versions/`](versions/) | one module per policy version, registered in `versions/__init__.py` |
| [`schema.py`](schema.py) | `PolicyCandidate`, `PolicyEvaluation`, `PolicyExclusion`, `PolicyOverride`, `EvalPolicy` |
| [`resolve.py`](resolve.py) | the settings each evaluation runs with for one candidate |
| [`cli.py`](cli.py) | the `policy` command |

A policy names its candidates, its evaluations and the window candidates are served at:

```python
POLICY = EvalPolicy(
    version="marin-2026-11-20-trial",
    serving_window=73_728,
    candidates=(QWEN, ...),
    in_distribution=(
        *defaults("boolq", "gpqa-diamond", "mrcr"),
        PolicyEvaluation("olympiadbench", judge="together", seed=42),
        PolicyEvaluation("mrcr", protocol="32k"),
        PolicyEvaluation("swebench-verified", protocol="mini-swe-agent", concurrency=16),
    ),
)
```

- A candidate is a [model catalog](../serve/models/README.md) entry pinned to a Hugging Face commit,
  with the largest context the model supports.
- An evaluation is a [benchmark catalog](../catalog.py) benchmark; `defaults(...)` lists those run with
  the benchmark's defaults. Its fields choose the variant (`protocol`, `num_fewshot`, `prompt_format`),
  name it (`label`, where the benchmark's own name won't do), and set what the policy runs it with
  (`max_output`, `seed`, `judge`, `concurrency`, `task_machine`). The variant owns prompts, thinking
  mode, trials, output caps and metrics. Construction fails unless the catalog declares the variant
  and a judge is named exactly when the benchmark needs one.
- A judge is named from the `judges:` entries of `configs/evaluations.yaml`.
- A `PolicyExclusion` drops one evaluation for one candidate; a `PolicyOverride` changes its output
  limit. Each needs a reason.

Each candidate is served at one window: the policy's `serving_window`, or the candidate's own window
if smaller. An evaluation's context depends on what its benchmark declares:

| Benchmark declares | Context |
| --- | --- |
| a protocol window, such as MRCR-32k's 32,704 | exactly that window; a candidate served below it must be excluded |
| an agent input limit, such as SWE-bench's 32,768 | the smaller of that limit and the serving window |
| nothing | the serving window |

Print a policy's resolved settings for every candidate:

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial [--candidate NAME]
```

Launch a candidate's evaluations, one job per server, without waiting for them:

```bash
uv run python -m experiments.evaluation.policy.cli launch --policy marin-2026-11-20-trial \
  --candidate qwen3.6-35b-a3b --priority batch [--evaluations a,b] [--limit 20] [--dry-run]
```

Each candidate runs on one server at its serving window, split only when its judges need different
keys, and the server runs several evaluations at once. A full run (no `--limit`) launches only from a
clean checkout at HEAD, so its records name the commit that ran.
