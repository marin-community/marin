# TaskTrove Clean

Converts [open-thoughts/TaskTrove](https://huggingface.co/datasets/open-thoughts/TaskTrove) into
tasks graded by one tool, [`tasktrove-verify`](../../../lib/tasktrove-verify/README.md), in one
declared mode each. A converted task keeps TaskTrove's gzip-tar layout: `instruction.md`,
`task.toml` (Harbor timeouts plus selection metadata), its own `environment/Dockerfile` with the
tool install appended, a three-line `tests/test.sh`, and `tests/verifier.toml` holding the mode
and its parameters. `contract.py` is the on-disk contract; the spec itself lives in the tool.

```
python -m experiments.post_training.tasktrove.pipeline --version 2026.09.09 --verify-tool-ref <sha> --run
```

`--verify-tool-ref` is the git ref of `lib/tasktrove-verify` written into every Dockerfile, so it
is part of every task's identity.

## Steps

| step | module | what it does |
|---|---|---|
| `raw` | `pipeline.py` | `hf_download` of every `*/tasks.parquet` at the pinned revision |
| `summaries` | `fingerprint.py` | fingerprint every task and `group_by` template id (normalized `tests/` and `environment/` code) |
| `templates` | `fingerprint.py` | `templates.json`, one exemplar per template with 20+ tasks, `coverage.json` per converter key |
| `converted` | `convert.py` | route by `source_verdicts.json` and converter key; emit the new binary and selection columns |
| `graded` | `verify.py` | `group_by` instruction within a source (lowest path wins; optional `--max-tasks-per-source`), then throw away tasks whose grader does not hold up; every row leaves with its final status |
| `clean` | `clean.py` | `tasks/part-*.parquet` (survivors only, copied in parallel), `ledger.parquet`, `manifest.json`, `report.md` |

The Zephyr stages load the source parquets with `load_parquet`, shuffle the rows into 1,024 even
shards before any per-task work (twenty sources store every task in a single row group), and use
`group_by` for the template index and dedup. The `templates` step and the ledger and manifest of
`clean` run on the coordinator over small columns only. `source_verdicts.json` keeps or drops each
source with a reason; every kept source has a converter in this tree.

Output rows carry `source`, `family`, `template_id`, `converter`, `mode`, `dockerfile_id`,
`language`, `tags`, `has_solution`, `task_binary`, and `solution_binary`. The oracle solution
never ships inside the binary the agent sees. `mode` says how a task is graded; `tags` are the
converter's selection labels, counted in `manifest.json` and `report.md` under `by_tag`. Judge-graded
tasks carry `judge` plus `reference` (an exact gate over gold answers, then the judge) or
`rubric` and `no-reference` (a checklist with no gold answer anywhere; the reward is the judge's
reading of the rubric), so a mix can include or exclude the rubric-only tasks by tag.
`python -m experiments.post_training.tasktrove.clean export <tasks-dir> <path>` writes one task
back out as a Harbor task directory; `clean summary` rewrites the ledger, manifest and report of
an existing output.

## Converters

A converter is selected by a `ConverterKey`: the source family from `source_verdicts.json` plus
the set of code files under `tests/` in the original template. `coverage.json` in the `templates`
step lists every key over the kept sources with its task count, exemplar template, and whether a
converter is registered; the `converted` step refuses to run while a key with an exemplar-sized
template is uncovered.

To write one:

1. Find the key in `coverage.json` and open `templates/<exemplar_template>/exemplar/`: the old
   `tests/test.sh`, verifier code, and the per-task data files.
2. Write `convert_<name>(task: TaskFiles) -> ConvertedTask | Rejected` in a module under
   `converters/`, following `converters/nemotron_gym.py`. Read the per-task data files, choose one
   mode, pass `instruction.md` and the Dockerfile through (edit the Dockerfile only to remove
   dependencies the old grader needed), set `tags` and `language`. Return `Rejected` with a
   `ConvertStatus` for a task the template cannot grade soundly.
3. Register a `Converter(name, keys, convert)` in the module and add it to `registry.py`.
4. Check in the exemplar as `fixtures/<name>.tar.gz` and add a test under `tests/`.
5. Run the sample harness against the local parquet and commit its report under
   `converters/reports/<name>.json`:

   ```
   uv run python -m experiments.post_training.tasktrove.sample_run --source <source> --count 20 --out /tmp/sample
   ```

   It builds each distinct Dockerfile with the local tool checkout, runs the shim on an empty
   workspace (must score 0) and, where a solution ships, after `solution/solve.sh` (must score 1).
   Containers run without a network unless `--network bridge` is passed (SWE oracles clone and
   install). Answer modes ship a synthesized `solution/solve.sh` that writes the expected value,
   so the oracle check covers them too. The tests never need Docker: they run the tool
   in-process on the checked-in fixtures.

Stdio converters keep a task only when at least one hidden input is absent from the prompt: a
task whose only cases are the samples in the problem statement is solved by printing the sample
outputs. Case count alone is not the signal (most one-case codeforces tasks hold an unseen input;
most one-case TACO and code-contests tasks are samples).

## Verification

`verify.py` runs seven checks in cost order: the spec parses and its files exist; the Dockerfile
carries the install block and nothing from the old grader; no expected value or `solution/` is
visible to the agent; the per-mode shape holds; and for output-file modes the tool is run
in-process on an empty output (0), the expected value (1), and a perturbation (0).

## Validity sample

`validity.py` asks whether a capable model can solve the tasks, so a group nobody can solve stands
out as a broken environment rather than a hard one. `validity sample` draws a stratified sample
(per converter, mode or source) from the clean parquet as task directories; `validity solve` sends
each instruction and Dockerfile to a headless agent (`claude -p --model sonnet` by default) and
keeps the one bash script it replies with as `candidate/solve.sh`; `validity_daytona.py` grades the
empty workspace, the oracle and the candidate in fresh Daytona sandboxes, one snapshot per distinct
Dockerfile; `validity report` tabulates the verdicts per group. The Daytona runner is a standalone
script (`uv run --no-project --isolated --prerelease=allow --with "daytona>=0.182,<1"`) because the
Daytona SDK does not resolve against this project's lock. Judge-graded tasks need
`TASKTROVE_JUDGE_*` forwarded with `--env`; without an endpoint their candidate check is an
infrastructure error, not a score.
