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
  --prompt-file scripts/gdn/codex_iteration_prompt.md \
  --post-check "uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both"
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
- Ensure the prompt enforces aggressive optimization strategy. `scripts/gdn/codex_iteration_prompt.md` now requires:
  - 3-candidate shortlist each iteration,
  - one high-upside structural choice,
  - no standalone scalar-only tuning iterations,
  - escalation after low-impact (<3%) results.
- If TPU queueing is unstable, switch profile runs to `dev-tpu-profile` on an allocated TPU.
- If runs flap on infra errors, restart from the latest successful commit.

## Stop/Resume
- Stop: Ctrl-C.
- Resume: rerun the same command; the next iteration starts from current `HEAD`.
- If a run failed after creating a bad commit, fix/revert manually before restarting.
