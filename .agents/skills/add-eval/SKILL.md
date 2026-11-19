---
name: add-eval
description: Add or change a non-agentic benchmark in Marin's curated evaluation catalog, then validate it through tests, a local corpus build, a reference comparison, and a small-model smoke run. Use for new likelihood, generation, or LLM-judged benchmarks and for prompt, scorer, or protocol changes to existing ones; Harbor agentic benchmarks are out of scope.
---

# Add a curated evaluation

Follow [Add an evaluation benchmark](../../../docs/tutorials/add-evaluation.md)
for code locations, patterns and commands. This skill adds the decisions,
evidence and stopping rules an agent needs.

## Before writing code

1. Search for the benchmark in `experiments/evaluation/benchmarks/` and with
   `uv run python -c "from experiments.evaluation.catalog import evaluation_benchmarks; print(sorted(evaluation_benchmarks()))"`.
   Search open PRs and issues too (`gh pr list --search <name>`); several
   benchmarks have in-flight integrations.
2. Identify the reference implementation and record it with a commit or version:
   lm-eval-harness task YAML, the paper's released runner, or a model card's
   evaluation code. The prompt, answer extraction, shots, sampling and split
   come from it. If no reference exists, state that and choose each setting
   explicitly.
3. Identify the reference score you will validate against: a per-item run of
   the reference implementation, or a published score for a model Marin can
   serve. Prefer a per-item comparison.
4. Pick the pattern from the guide's table and read its example module end to end.
5. Confirm the hardware you may launch on. GPU smoke runs use CoreWeave and need
   `CW_KEY_ID` and `CW_KEY_SECRET` in your environment; the launcher refuses
   them otherwise. Without the keys, the model's default GCP TPU needs none.
   When delegating, state which platforms the delegate may use so a missing
   credential does not stop the work. Never search for or copy credentials
   that the requester did not provide.

Stop and report when the source is gated or licensed in a way that forbids
evaluation use, or when the reference requires tools or multi-turn
interaction (that is a Harbor benchmark).

## Implement

- Pin the dataset commit and exact files. Count the evaluated rows from the
  pinned files; never copy a population from a paper.
- Keep sampling at the served model's defaults unless the reference fixes it;
  fixed values apply to every model evaluated.
- Put prompt rendering, conversion and scorers in `lib/taskcompendium`; put
  pins and protocols in `experiments/evaluation/benchmarks/`.
- Set the protocol `benchmark` to the intended `--evals` key.
- Bump the recipe and protocol versions whenever prompt or conversion code
  changes; corpus identity does not hash that code. Changing an existing
  `multiple_choice_evaluation` template means bumping `EXPLICIT_TEMPLATE_VERSION`,
  which republishes every benchmark that helper builds; say so in the PR.
- Do not add the benchmark to a suite in `configs/evalchemy.yaml` unless the
  user asks.

## Validate

Run every step; each produces evidence for the PR.

1. **Tests.** Prompt rendering against a prompt copied from the reference's
   output, and scorer cases for correct, incorrect, unparseable and ambiguous
   answers. Run the affected tests and `./infra/pre-commit.py --fix` on the
   changed files.
2. **Local corpus build.** Build the full population with the curation driver
   and compare every compiled request character by character with the
   reference's rendering of the same rows.
3. **Smoke.** Dry-run, then launch with `--model qwen3-0.6b --limit 50
   --version dev`, adding `--platform gpu --accelerator H100x1
   --federated_cluster cw-us-east-02a` when CoreWeave keys are available.
   Follow `use-iris` for job monitoring with a bounded wait. Confirm with
   `cli scores` that every selected item is scored and no infrastructure
   errors remain.
4. **Full-population run and reference comparison.** Run without `--limit` on
   the same model as the reference. For a per-item reference, report the
   agreement rate and explain each disagreement. For a published score, report
   both values with the standard error and name every known protocol
   difference.
5. **Grader spot check.** For generation benchmarks, read at least 20 graded
   responses from the archive, including every unparseable answer. For an LLM
   judge, use `audit-grader`.

A result outside two standard errors of the reference, or item agreement below
about 98% for likelihood benchmarks, is a bug until explained. Fix prompt or
extraction differences before publishing; do not tune toward the reference
number.

## Hand off

Publish through `manage-pr`. The PR body states the source pin and license,
the reference implementation, the protocol choices, and a validation table:
model, population, score, standard error, coverage, reference value and
agreement. Link the run records with `scores` output. Clean up local corpus
builds and any reference-harness environments created for the task.
