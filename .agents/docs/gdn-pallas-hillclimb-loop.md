# GDN Pallas Hill-Climb Loop

Run repeated Codex iterations that each produce one validated TPU optimization commit.

Target outcome:
- Move quickly toward major speedups (not fractional tuning gains), with MFU trajectory aiming for ~50%.
- Require structural kernel/algorithm changes when current traces show the same dominant hotspots.

## One-Time Setup
1. Ensure local auth/env is ready:
   - `RAY_AUTH_MODE=token`
   - `HF_TOKEN`
   - `WANDB_API_KEY`
2. Start from a clean branch (recommended prefix: `codex/gdn-...`).
3. Confirm helper CLI works:
   - `uv run python scripts/gdn/gdnctl.py --help`

## Autonomous Loop Command

```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 10 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --resilient \
  --directive-preset triangular-inversion \
  --dirty-policy stash \
  --no-commit-policy count-failure \
  --prompt-file scripts/gdn/codex_iteration_prompt.md \
  --hold-dev-tpu \
  --dev-tpu-cluster us-central1 \
  --dev-tpu-fallback-cluster us-east5-a \
  --dev-tpu-name "$USER-gdn" \
  --dev-tpu-type v5p-8 \
  --dev-tpu-allocate-attempts 2 \
  --dev-tpu-allocate-retry-sleep 20 \
  --post-check "uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name $USER-gdn --tests both" \
  --post-check "uv run python scripts/gdn/gdnctl.py lint-log"
```

## How The Loop Behaves
For each iteration:
1. Verifies working tree cleanliness (unless `--allow-dirty`).
2. Runs `codex exec` non-interactively with the iteration prompt.
3. Requires a new commit (unless `--allow-no-commit`).
4. Runs all `--post-check` commands.
5. Proceeds to next iteration or stops on failure.

Prompt and final-response logs are stored under:
- `.agents/logs/gdn_codex_loop/`

## Operational Advice
- Prefer `--post-check` commands that are strict but fast enough to run every iteration.
- Keep each iteration scoped to one optimization so regressions are easy to bisect.
- Prefer `--resilient` for unattended runs; it keeps the loop alive through transient command/network/allocation failures.
- Ensure the prompt enforces aggressive optimization strategy. `scripts/gdn/codex_iteration_prompt.md` now requires:
  - 3-candidate shortlist each iteration,
  - one high-upside structural choice,
  - no standalone scalar-only tuning iterations,
  - escalation after low-impact (<3%) results.
- Prefer `.json.gz` traces by default and only download `.xplane.pb` artifacts when doing XProf analysis.
- `--directive`, `--directive-file`, and `--directive-preset` allow per-session research directives without editing shared prompt files.
- Preset directives are backed by markdown docs in `scripts/gdn/session_directives/` (for example `triangular-inversion`).
- `--dirty-policy stash` avoids hard stops after failed attempts leave a dirty tree.
- `--dirty-policy stash` restores the stashed tree automatically after each iteration.
- If stash restore conflicts with newly-generated iteration edits, default `--stash-restore-policy warn-keep` keeps the stash and continues; set `--stash-restore-policy fail` to stop instead.
- `--no-commit-policy count-failure` allows the loop to continue when an iteration intentionally records a failed attempt without a commit.
- `--hold-dev-tpu --dev-tpu-name <name>` keeps a dev TPU allocation active for the whole loop and releases it automatically on exit.
- Keep `--post-check` cluster/name aligned with the active held allocation (`--cluster` + `--tpu-name`) so checks run on the right TPU.
- If TPU queueing is unstable, switch profile runs to `dev-tpu-profile` on an allocated TPU.
- If runs flap on infra errors, restart from the latest successful commit.

## Stop/Resume
- Stop: Ctrl-C.
- Resume: rerun the same command; the next iteration starts from current `HEAD`.
- If a run failed after creating a bad commit, fix/revert manually before restarting.
