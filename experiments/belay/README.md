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
| `queued` | accepted for testing (an approved intake comment, or a maintainer or agent proposal); no trial yet |
| `running` | at least one trial in flight |
| `done` | trials finished, no decision yet |
| `adopted` | **in the current checkpoint recipe** (today Belay Ckpt 1, `marin_belay_v1_ckpt1`) |
| `removed` | adopted at some point, not in the current checkpoint |
| `closed` | decided against |

`adopted` always means "in the latest checkpoint": when a new checkpoint drops a feature, its entry gets a `removed`
event naming the checkpoint, and a feature added by the checkpoint gets an `adopted` event.

## Where things live

- **Ids**: assigned in order when an idea is approved (the intake bot does this for #9514 comments) and never reused.
- **Code**: an experiment's code is a default-off flag on its batch branch, `marin_belay_v1_b<batch>`. The trial
  records the exact `commit` and its `settings` (the `--model-set` / `--opt-set` flags), so commit + settings
  reproduce it. A lightweight tag `belay/<id>` may point at that commit; there is no branch per experiment.
- **Checkpoints**: an adopted flag is folded into the next `marin_belay_v1_ckpt<N>` branch, which holds only that
  checkpoint's code.
- **Runs**: W&B runs (project `marin_moe`) are listed in each trial's `wandb`; new runs carry the id in their name.
- **Discussion**: batch plans and results stay in the batch's comment on #9451; trials link to it (`comment_url`).

## Proposing an idea

Comment on the weekly intake issue ([marin#9514](https://github.com/marin-community/marin/issues/9514)) with the
template in [`intake_template.md`](intake_template.md). A 👍 from a maintainer approves it: the intake bot copies it
into #9451, assigns the next id and writes a `status: queued` entry with the comment's text, which the Belay agent
then fills in as it runs trials.

## Entry format

Required: `id`, `slug`, `name`, `category` (one of `CATEGORIES` in `registry.py`), `summary`, `source`
(`intake`, `maintainer`, `agent`, `community`, `baseline`), `status`. Optional: `first_batch`, `tags`, `papers`
(`arxiv` and/or `title`), `details`, `knobs`, `intake` (`url`, `author`, `approved_by`; required for
`source: intake`), `trials` and `events`. A trial has `id`, `variant`, `rung`, `protocol`, `batch`, `date`, `comment_url`
and optionally `baseline`, `steps`, `delta` and `loss` (one per seed), `wandb`, `settings`, `commit`,
`step_time_change`, `tuned`, `note`, `derived`.
A trial id is `t001`, `t002`, ... in run order, unique within its experiment and never reused; cite a trial elsewhere as `<experiment id>.<trial id>`, e.g. `X0014.t005`.
An event is `adopted`, `removed` or `closed`, with `batch`, `date`,
`comment_url` and optionally `note`, `candidate`, `as_adopted`.
