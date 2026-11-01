Run one scheduled Marin Belay pass.

1. Check out `marin_belay_v1_recipe` and run
   `uv run python -m experiments.belay start --check-only`.
2. If it prints `no work`, end the session with that line. Do nothing else.
3. If it prints `work:`, run `uv run python -m experiments.belay claim` and
   follow `experiments/belay/RUNBOOK.md` from section 0 with the events file it
   prints. End with section 7, then report it with the `finish` command that
   `claim` printed.
