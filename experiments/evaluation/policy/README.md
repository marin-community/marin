# Eval policies

An eval policy says which models run which benchmarks, with which settings, at which window. Marin's policy
is tracked in [#10197](https://github.com/marin-community/marin/issues/10197).

| File | Holds |
| --- | --- |
| [`versions/`](versions/) | one directory per policy version, registered in `versions/__init__.py` |
| [`schema.py`](schema.py) | `Policy` and the checks it makes when built |
| [`launch.py`](launch.py) | the models and evaluations a launch runs, and their servers |
| [`runs.py`](runs.py) | finding a policy's runs and checking them |
| [`cli.py`](cli.py) | the `policy` command |

## Declaring a policy

Each version is a directory, `versions/<version>/`, holding three files:

- `policy.py` builds the `Policy`.
- `evaluations.yaml` names its evaluations, in the format of
  [`configs/evaluations.yaml`](../configs/evaluations.yaml).
- A README lists how it differs from earlier policies.

```python
POLICY = Policy(
    version="marin-2026-11-20-trial",
    models=("muse-glimmer-30b", "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16", "qwen3.6-35b-a3b"),
    evaluations=load_evaluations(Path(__file__).with_name("evaluations.yaml")),
    serving_window=73_728,
    exclusions={},  # {(model, evaluation): reason}
)
```

```yaml
evaluations:
  aime24:
    benchmark: aime24
    run:
      seed: 42
  mrcr-32k:
    benchmark: mrcr
    protocol: 32k
```

- **Models** are [model catalog](../serve/models/README.md) keys. Each must pin a Hugging Face commit for
  its weights and its tokenizer.
- **Evaluations** are [benchmark catalog](../catalog.py) variants, named by their keys:
  - `protocol`, `num_fewshot` and `prompt_format` choose the variant, which owns prompts, thinking mode,
    trials, output caps and metrics;
  - `run` sets what the policy runs it with: `seed`, `max_tokens`, the judge, `concurrency`, `task_machine`.
- **Construction fails** if:
  - the catalog doesn't declare a variant;
  - an evaluation that needs a judge doesn't name one, or one that doesn't need a judge names one;
  - an evaluation sets an item limit;
  - a name repeats;
  - an exclusion names a model or evaluation the policy doesn't have.
- **A version must not start with `eval-policy-`**; that prefix names policy tags.

## Windows

Each model is served at one window: the policy's `serving_window`, or the model's own if smaller. A model's
own window is its serving config's `max_model_len`, else the one its checkpoint declares. An evaluation's
context depends on what its benchmark declares:

| Benchmark declares | Context |
| --- | --- |
| a protocol window, such as MRCR-32k's 32,704 | exactly that window |
| an agent input limit, such as SWE-bench's 32,768 | the smaller of that limit and the serving window |
| nothing | the serving window |

A policy model served below an evaluation's protocol window is an error unless the policy excludes that pair.

## Running

```bash
uv run python -m experiments.evaluation.policy.cli show --policy marin-2026-11-20-trial [--model NAME]
uv run python -m experiments.evaluation.policy.cli launch --policy marin-2026-11-20-trial \
  --all-models --priority batch [--evaluations a,b] [--limit 20] [--dry-run]
uv run python -m experiments.evaluation.policy.cli status --policy marin-2026-11-20-trial
uv run python -m experiments.evaluation.policy.cli verify --policy marin-2026-11-20-trial RUN_ID...
```

`launch` and `status` take the models to run in one of three ways:

| Option | Runs |
| --- | --- |
| `--model KEY`, repeatable | the named catalog models; a key outside the policy runs as a supplied model |
| `--all-models` | every model of the policy (the default for `status`) |
| `--model-config FILE` | a supplied model, such as a training checkpoint |

A supplied model leaves out each evaluation it can't run, such as one whose protocol window is larger than its
own; the dry run and `status` list them. A supplied model may not take a catalog model's name.

Each model runs on one server, split only when its judges need different keys, and the server runs four
evaluations at once. A full run (no `--limit`) launches only from a clean checkout. `launch`, `status` and
`verify` need CoreWeave credentials.

## Runs

A run's ID is `<base>-<commit>[-dirty<digest>]-a<attempt>`:

- **The base** digests the run's inputs: the policy version, the model, the evaluation and its settings, the
  evaluator runtime, the judge, the server's vLLM build and accelerator, an object-store checkpoint's file
  checksums, and an agent's concurrency.
- **The commit** is the one the launch ran from.
- **A dirty checkout** adds a digest of its uncommitted changes. Such a run, allowed only for smoke tests, never
  complies, and a clean launch never resumes it.
- **The attempt** starts at 1.

A run resumes only at its own commit and attempt, so its attempts never mix code. Runs are never deleted or
overwritten.

A run complies if:

- it was launched from a clean checkout;
- its record succeeded and its step finished successfully;
- its model, evaluation and runtime match the policy's;
- its archive shows the window it was served at, and, for a full run, every item decided. A full run may leave
  out items its protocol excludes, such as ones too long for its window, but not items it never inspected.

A ⚠ marks a compliant run whose replies often hit their output cap or whose attempts often went unscored.

`launch` leaves out an evaluation that has a compliant run with the same inputs at any commit, so each pass
runs only what is missing, broken or changed. To rerun a fixed evaluation, bump its benchmark's `VERSION`,
which changes its inputs. A run at the launch's commit that finished but doesn't comply is listed, not rerun:
`--rerun RUN_ID` runs it again as a new attempt beside the old one. `--rerun` takes only a run that is invalid
or flagged, so a valid run can't be redrawn.

`status` shows each evaluation's compliant run and the commit it ran at, or else its newest run and why it
doesn't comply. `verify` checks runs from a clean checkout of their own commit, names each problem, and exits
non-zero unless every run complies.
