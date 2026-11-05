Run one scheduled Marin Belay pass.

1. Check out `marin_belay_v1_recipe` and run
   `uv run python -m experiments.belay start --check-only`.
2. If it prints `no work`, end the session with that line. Do nothing else.
3. If it prints `work:`, run `uv run python -m experiments.belay claim`, then
   follow `experiments/belay/RUNBOOK.md` with the events file it prints: work
   through the events, then the drain loop (`wait --session <id>`) until it
   prints `drained:`. End with section 7, then report it with the `finish`
   command that `claim` printed.
