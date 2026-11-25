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

Launch a candidate's evaluations, one job per server, without waiting for them:

```bash
uv run python -m experiments.evaluation.policy.cli launch --policy marin-2026-11-20-trial \
  --candidate qwen3.6-35b-a3b --priority batch [--evaluations a,b] [--limit 20] [--dry-run]
```

Evaluations share a server when they need the same window. Agent and single-turn evaluations never
share one, nor do evaluations whose judges need different keys. A full run (no `--limit`) launches
only from a clean checkout at HEAD, so its records name the commit that ran.

A run's ID comes from the policy version and the run's inputs, so launching again finds the same
paths: it skips runs whose record succeeded and reruns ones that failed. A run that succeeded but is
invalid (for example, another server window) is therefore not rerun; `status` says so for each such
run.

Check the runs a launch from this checkout would make, or any runs by ID:

```bash
uv run python -m experiments.evaluation.policy.cli status --policy marin-2026-11-20-trial
uv run python -m experiments.evaluation.policy.cli verify --policy marin-2026-11-20-trial RUN_ID...
```

`status` shows each evaluation's full and smoke run: valid, invalid, not run, or differing from this
checkout (run at another commit with other inputs). A ⚠ marks a run
whose replies often hit their output cap or whose attempts often went unscored. `verify` checks any
run against the policy and names each field that differs. A full run must decide every item: it may
leave out items its protocol excludes, such as ones too long for its window, but not items it never
inspected. `launch`, `status` and `verify` need CoreWeave credentials.

A policy marked `released` launches only from a clean checkout of its tag, `eval-policy-<version>`,
and `status` and `verify` mark its runs from any other commit invalid. Its run IDs include the tag's
commit, so runs made before the release, at the same version, never stand in for the release's. A
fix to a released policy is a new version. A policy not yet released launches from any commit.
