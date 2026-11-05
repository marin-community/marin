# Marin Belay

Marin Belay (marin#9451) tests one idea at a time against the baseline its Track names (Vanilla, Basic MoE or the
current recipe at a rung, or the 8-minute speedrun) and records every result, good or bad, in the experiment
registry. The current recipe is Belay Ckpt Oct 20 2026: branch `marin_belay_v1_recipe`, tag
`belay-ckpt-2026-10-20`, launched with `belay_s7.py --ladder s7n`.

**Agent sessions follow [`RUNBOOK.md`](RUNBOOK.md) exactly.** The runbook and its commands
(`uv run python -m experiments.belay <section>`) are the procedure. [`config.py`](config.py) holds every fixed
value: branches, cluster, priority, node cap, and each Track's baseline, metric and noise band. Nothing else restates
them.

## Branches, issues and edits (for people)

- **Which branch to build on.** The recipe is `marin_belay_v1_recipe`. Only a maintainer's `recipe-change` moves
  it, and tags each new tip `belay-ckpt-<YYYY-MM-DD>`.
  - Experiment code goes on this week's branch, `marin_belay_v1_w<monday>`. The week's first session cuts it fresh
    from the recipe with `weekly-rollover`, carrying the code of ideas still in play (and porting any that no
    longer apply). After that, `start` checks it out and merges in new recipe commits by itself.
  - In the demo the weekly branches are `marin_belay_v1_loomdemo_w<monday>`, cut from `marin_belay_v1_loomdemo`.
  - Never base work on `main`, `marin_belay_v1_ckpt*`, `marin_belay_v1_b<batch>`, `ladderclimb_*` or the GB200
    lineages.
- **How ideas arrive.**
  1. Anyone comments on marin#9514, one idea per comment, optionally with a `Track:` line (default `speedrun`;
     `ladder` for every competitive rung; RUNBOOK.md section 9). The approver's instructions may set it instead.
  2. A maintainer approves it on the Belay admin page, which reads #9514 live and shows the visible text it will
     store (hidden HTML and invisible characters stripped, and flagged). GitHub reactions approve nothing.
  3. The session reads only that stored text (an edit on GitHub after approval is refused), gives the idea its
     registry id. Nothing is posted on #9514: `sync` adds 👀 to a suggestion once it is approved or declined, and
     results, and declines with the approver's reason, are on the public Status page.
  - Plans, stage tables and verdicts go to the page's agent log, one thread per idea at `#status/X####` on the page, updated in
    place. Nothing new is posted on #9451; its existing comments stay as they are.
- **How edits are made.**
  - A session changes nothing by hand, except resolving a weekly-rollover port (RUNBOOK.md section 4). Each
    command writes, commits and pushes its own files:
    - the registry entry (on the registry branch);
    - the run spec `experiments/belay/runs/X####.yaml`, or one per ladder rung (`X####d768.yaml`), on the weekly
      branch;
    - the trial and its results (on the registry branch).
  - An idea that needs new code gets a new default-off config field. `code-change --finish` commits it straight to
    the weekly branch once its fixed checks pass (RUNBOOK.md section 1b); no human review gates it. The idea then
    runs as configuration. Flags that never win stay as dead default-off code until a maintainer deletes them.
- **The recipe** changes only when maintainers accept a checkpoint (runbook section 5).
- **The page** shows the registry. `B sync`, which the Loom watch runs every pass, uploads `registry.py build`
  along with run states and storage; a maintainer releases the app from main.

## Where things are

| File | What it holds |
|---|---|
| `RUNBOOK.md` | The session procedure, one section per command |
| `__main__.py` | The commands, one function per runbook section |
| `config.py` | Every fixed value, and the Track table |
| `branch.py` | Weekly branch and git checks |
| `code_change.py` | The checks that gate a code change |
| `registry_entry.py` | Writes to the registry checkout |
| `submit.py` | Run specs, launch checks, the launch command |
| `results.py` | W&B results and the win/noise/loss rule |
| `page.py` | The Belay page API client |
| `demo.py` | Local demo only: Marina, watcher, dispatcher, sessions |
| `runs/X####.yaml` | Run specs, on weekly branches |

The registry itself is on the registry branch, at `experiments/belay/registry/X####-<slug>.yaml`; its format is
in that branch's `README.md`.
