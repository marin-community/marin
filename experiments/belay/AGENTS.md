# Marin Belay

Marin Belay (marin#9451) tests one idea at a time against the current recipe, Belay Ckpt Oct 20 2026
(`belay_s7.py --ladder s7n` on `marin_belay_v1_clean_s7`), on the d512 rung, and records every result, good or
bad, in the experiment registry.

**Agent sessions follow [`RUNBOOK.md`](RUNBOOK.md) exactly.** The runbook and its commands
(`uv run python -m experiments.belay <section>`) are the procedure. [`config.py`](config.py) holds every fixed
value: branches, cluster, priority, node cap, metric and noise threshold. Nothing else restates them.

## Branches, issues and edits (for people)

- **Which branch to build on.** The recipe is `marin_belay_v1_clean_s7`; never commit to it.
  - Experiment code goes on this week's branch, `marin_belay_v1_w<monday>`. `weekly-rollover` cuts it from last
    week's branch, merges the base branch in and pushes it, the first time a session runs that week.
  - In the demo the weekly branches are `marin_belay_v1_loomdemo_w<monday>`, cut from `marin_belay_v1_loomdemo`.
  - Never base work on `main`, `marin_belay_v1_ckpt*`, `marin_belay_v1_b<batch>`, `ladderclimb_*` or the GB200
    lineages.
- **How ideas arrive.**
  1. Anyone comments on marin#9514.
  2. A maintainer's 👍 makes the intake bot (every 5 minutes) copy the comment into #9451 and add 🚀.
  3. The approval reaches a session as an event, and the session gives the idea its registry id.
  - There is one running issue, #9451; there are no weekly issues.
  - In the demo, approvals come from the admin page's Approve button.
- **How edits are made.**
  - A session changes nothing by hand. Each command writes, commits and pushes its own files:
    - the registry entry (on the registry branch);
    - the run spec `experiments/belay/runs/X####.yaml` (on the weekly branch);
    - the trial and its results (on the registry branch).
  - An idea that needs new code is escalated. A maintainer then adds a default-off `--model-set` / `--opt-set`
    field on the weekly branch, after which the idea runs as configuration.
- **The recipe** changes only when maintainers accept a checkpoint (runbook section 5).
- **The page** shows the registry. The watcher uploads `registry.py build`, and `demo release` re-releases the
  app.

## Where things are

| File | What it holds |
|---|---|
| `RUNBOOK.md` | The session procedure, one section per command |
| `__main__.py` | The commands, one function per runbook section |
| `config.py` | Every fixed value |
| `branch.py` | Weekly branch and git checks |
| `registry_entry.py` | Writes to the registry checkout |
| `submit.py` | Run specs, launch checks, the launch command |
| `results.py` | W&B results and the win/noise/loss rule |
| `page.py` | The Belay page API client |
| `demo.py` | Local demo only: Marina, watcher, dispatcher, sessions |
| `runs/X####.yaml` | Run specs, on weekly branches |

The registry itself is on the registry branch, at `experiments/belay/registry/X####-<slug>.yaml`; its format is
in that branch's `README.md`.
