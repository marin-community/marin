# Eval policies

An eval policy says which models (candidates) run which benchmarks, with which settings, and where
a candidate departs from them. Each version is a YAML file, `versions/<version>/policy.yaml`, beside a
README that lists its candidates, its evaluations and how it differs from earlier policies. Marin's
policy is tracked in [#10197](https://github.com/marin-community/marin/issues/10197).

| File | Holds |
| --- | --- |
| [`versions/`](versions/) | one directory per policy version; `versions/__init__.py` loads every `policy.yaml` |
| [`load.py`](load.py) | reads a policy file into the schema, rejecting unknown keys and names |
| [`schema.py`](schema.py) | `PolicyCandidate`, `PolicyEvaluation`, `PolicyExclusion`, `PolicyOverride`, `EvalPolicy` |
| [`resolve.py`](resolve.py) | the settings each evaluation runs with for one candidate |
| [`cli.py`](cli.py) | the `policy` command |

A policy file names everything it uses:

```yaml
version: marin-2026-11-20-trial
default_context: 73728
candidates:
  qwen3.6-35b-a3b: {max_context: 262144}
evaluations:
  - boolq
  - {benchmark: olympiadbench, judge: together, seed: 42}
  - {benchmark: mrcr, protocol: 32k}
  - {benchmark: swebench-verified, protocol: mini-swe-agent, concurrency: 16}
exclusions:
  - {candidate: qwen3.6-35b-a3b, evaluation: boolq, reason: ...}
overrides:
  - {candidate: qwen3.6-35b-a3b, evaluation: mrcr-32k, max_output: 4096, reason: ...}
```

- A candidate is a [model catalog](../serve/models/README.md) key, pinned there to a Hugging Face
  commit, with the largest context the model supports.
- An evaluation is a [benchmark catalog](../catalog.py) benchmark, alone when it runs with the
  benchmark's defaults. Its fields choose the variant (`protocol`, `num_fewshot`, `prompt_format`),
  name it (`label`, where the benchmark's own name won't do), and set what the policy runs it with
  (`context`, `max_output`, `seed`, `judge`, `concurrency`, `task_machine`). The variant owns prompts,
  thinking mode, trials, output caps and metrics.
- A judge is named from the `judges:` entries of `configs/evaluations.yaml`.
- An exclusion drops one evaluation for one candidate; an override changes its `context` or
  `max_output`. Each names the candidate and the evaluation, and needs a reason.

An evaluation's context depends on what its benchmark declares:

| Benchmark declares | Context |
| --- | --- |
| a protocol window, such as MRCR-32k's 32,704 | exactly that window; a candidate served below it must be excluded |
| an agent input limit, such as SWE-bench's 32,768 | the smallest of that limit, the requested context and the candidate's served window |
| nothing | the smaller of the requested context and the candidate's served window |

The requested context is the candidate's override, else the evaluation's `context`, else the policy's
`default_context`.

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
