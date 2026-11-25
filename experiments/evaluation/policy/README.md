# Eval policies

An eval policy says which models (candidates) run which benchmarks, with which settings, and where
a candidate departs from them. Each version is a Python module under [`versions/`](versions/), with a
README that lists its candidates, its evaluations and how it differs from earlier policies. Marin's
policy is tracked in [#10197](https://github.com/marin-community/marin/issues/10197).

| File | Holds |
| --- | --- |
| [`schema.py`](schema.py) | `PolicyCandidate`, `PolicyEvaluation`, `PolicyExclusion`, `PolicyOverride`, `EvalPolicy` |
| [`resolve.py`](resolve.py) | the settings each evaluation runs with for one candidate |
| [`cli.py`](cli.py) | the `policy` command |
| [`versions/`](versions/) | one module per policy version, registered in `versions/__init__.py` |

A candidate is a [model catalog](../serve/models/README.md) entry pinned to a Hugging Face commit,
with the largest context it supports. An evaluation names a [benchmark catalog](../catalog.py)
protocol, which owns prompts, thinking mode, trials, output caps and metrics, and the settings the
policy runs it with:

```python
PolicyEvaluation("olympiadbench", judge=JUDGES["together"], seed=42)
PolicyEvaluation("mrcr", protocol="32k")
PolicyEvaluation("swebench-verified", protocol="mini-swe-agent", concurrency=16)
```

An evaluation's context is the window its benchmark declares, else the policy's `context`:

| Benchmark declares | Context |
| --- | --- |
| a protocol window, such as MRCR-32k's 32,704 | exactly that window; a candidate served below it must be excluded |
| an agent input limit, such as SWE-bench's 32,768 | that limit, or less through an override, since the agent gets the smaller of it and the served window |
| nothing | the policy's `context`, or the evaluation's own |

Judges come from `evals.JUDGES`, the judges named in `configs/evaluations.yaml`. A
`PolicyOverride` changes a candidate's context or output limit on one evaluation; a
`PolicyExclusion` drops one. Each needs a reason.

Print a policy's resolved settings for every candidate:

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial [--candidate NAME]
```
