# Marin Belay experiment registry

Every idea tested in Marin Belay ([marin#9451](https://github.com/marin-community/marin/issues/9451)) has a permanent
id (`X0001`, `X0002`, ...) and one YAML file, `registry/<id>-<slug>.yaml`, on the `marin_belay_v1_registry` branch.
The file says what the idea is, where it came from, every trial of it and every decision taken on it. The Belay
page's Results section is built from these files.

```bash
uv run python -m experiments.belay.registry validate          # check every entry
uv run python -m experiments.belay.registry build > results.json   # the page's results data
uv run python -m experiments.belay.registry new --slug ... --name ... --category ... --summary ... --source agent
```

## Lifecycle

`status` is one of:

| status | meaning |
|---|---|
| `approved` | accepted for testing (an approved intake comment, or a maintainer or agent proposal); no trial yet |
| `running` | at least one trial in flight |
| `done` | trials finished, no decision yet |
| `adopted` | **in the current recipe** (today Belay Ckpt Oct 20 2026, `marin_belay_v1_recipe`; see below) |
| `removed` | adopted at some point, not in the current checkpoint |
| `closed` | decided against |

`adopted` always means "in the current recipe": when a new checkpoint drops a feature, its entry gets a `removed`
event naming the checkpoint, and a feature added by the checkpoint gets an `adopted` event. The `adopted` entries
today were adopted into Belay Ckpt 1; the Oct 20 2026 checkpoint's `adopted` and `removed` events are not recorded
yet.

## Where things live

- **Ids**: assigned in order by the Belay session when it starts an approved idea
  (`uv run python -m experiments.belay new-idea`, runbook section 1), and never reused.
- **Code**: an experiment's code is a default-off flag on that week's experiment branch,
  `marin_belay_v1_w<monday>`, next to its run spec `experiments/belay/runs/<id>.yaml`. The trial records the exact
  `commit` and its `settings` (the `--model-set` / `--opt-set` flags), so commit + settings reproduce it. There is
  no branch per experiment. Older trials ran on batch branches `marin_belay_v1_b<batch>`; their commits still
  reproduce them.
- **Checkpoints**: an adopted flag is folded into the next recipe branch, which holds only that checkpoint's code
  (today `marin_belay_v1_recipe`).
- **Runs**: W&B runs (project `marin_moe`) are listed in each trial's `wandb`; new runs carry the id in their name.
- **Discussion**: plans and results are posted to the Belay page and to #9451; trials link to them (`comment_url`).

## Proposing an idea

Comment on the intake issue ([marin#9514](https://github.com/marin-community/marin/issues/9514)) with the template
in [`intake_template.md`](intake_template.md), one idea per comment. A 👍 from a maintainer approves it: the intake
bot (every 5 minutes) copies it into [marin#9451](https://github.com/marin-community/marin/issues/9451) and marks
your comment 🚀. The next Belay session then gives it the next id, writes its `status: approved` entry, and fills
the entry in as its trials run. There are no weekly issues.

## Entry format

Required: `id`, `slug`, `name`, `category` (one of `CATEGORIES` in `registry.py`), `summary`, `source`
(`intake`, `maintainer`, `agent`, `community`, `baseline`), `status`. Optional: `track` (the baseline the idea runs against, from the intake comment's `Track:` line; set on
every new entry), `first_batch`, `tags`, `papers`
(`arxiv` and/or `title`), `details`, `knobs`, `intake` (`url`, `author`, `approved_by`; required for
`source: intake`), `trials` and `events`. A trial has `id`, `variant`, `rung`, `protocol`, `batch`, `date`, `comment_url`
and optionally `baseline`, `steps`, `delta` and `loss` (one per seed), `wandb`, `settings`, `commit` (the git SHA it ran, at least 10 characters), `step_time_change`, `tuned`, `note`, `derived`.
A trial id is `t001`, `t002`, ... in run order, unique within its experiment and never reused; cite a trial elsewhere as `<experiment id>.<trial id>`, e.g. `X0014.t005`.
An event is `adopted`, `removed` or `closed`, with `batch`, `date`,
`comment_url` and optionally `note`, `candidate`, `as_adopted`.
