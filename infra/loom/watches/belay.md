Run one scheduled Marin Belay pass.

1. Check out `marin_belay_v1_recipe`, run
   `uv run python -m experiments.belay sync` (it re-queues the events of a
   session that ended without reporting, and keeps the Belay page's runs,
   results and storage current), then run
   `uv run python -m experiments.belay start --check-only`.
2. If it prints `no work`, end the session with that line. Do nothing else.
   If `sync` fails, still run the check; report the failure in the final line.
3. If it prints `work:`, run `uv run python -m experiments.belay claim`, then
   follow `experiments/belay/RUNBOOK.md` with the events file it prints: work
   through every event, end with section 7, then report it with the `finish`
   command that `claim` printed. Never wait for runs, and run every command in
   the foreground: this session ends as soon as you stop working, which kills
   any background loop or scheduled wake-up. A later pass handles the runs.
