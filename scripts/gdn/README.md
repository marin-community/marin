# GDN Optimization Helpers

This directory contains wrappers for iterative TPU optimization of
`lib/levanter/src/levanter/layers/gated_deltanet.py`.

## Main CLI

`gdnctl.py` exposes subcommands for TPU tests, profile jobs, trace download, and unattended Codex loops.

```bash
uv run python scripts/gdn/gdnctl.py --help
```

## Common Commands

Run correctness tests on Ray TPU:
```bash
uv run python scripts/gdn/gdnctl.py ray-test --cluster us-central1 --tpu auto --tests both
```

Submit lightweight profile run:
```bash
uv run python scripts/gdn/gdnctl.py ray-profile --cluster us-central1 --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-wait
```

Run lightweight profile on an allocated dev TPU:
```bash
uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster us-central1 --tpu-name "$USER-gdn" --tpu v5p-8 --size 130m --num-steps 20 --profile-start-step 2 --profile-num-steps 6 --batch-size 8 --no-sync
```

Wait for a profile job:
```bash
uv run python scripts/gdn/gdnctl.py ray-wait --cluster us-central1 <job_id> --show-logs
```

Download traces from HF:
```bash
uv run python scripts/gdn/gdnctl.py hf-download-trace --repo-id <org/repo> --path-prefix <path>
```

Download XProf payloads only when needed:
```bash
uv run python scripts/gdn/gdnctl.py hf-download-trace --repo-id <org/repo> --path-prefix <path> --include-xplane
```

Run unattended Codex loop:
```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 5 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --resilient \
  --directive-preset training-chunk-kernel-focus \
  --directive-preset control-structure-pivot \
  --directive-preset macro-coverage-pivot \
  --dirty-policy stash \
  --no-commit-policy count-failure
```

By default, `codex-loop` hides noisy `file update:` / `diff --git` blocks from Codex output. Use `--show-file-updates` to display them.
By default, `codex-loop` enforces a per-iteration TPU validation gate: tests + one profile run before the iteration can complete.
By default, `codex-loop` now runs `codex exec --ephemeral` so unattended loops do not flood the Codex app session/thread database. Use `--no-codex-ephemeral` only when you intentionally want loop sessions persisted for manual debugging/resume.

To allocate and hold a dev TPU for the whole loop session:
```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 5 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --resilient \
  --hold-dev-tpu \
  --dev-tpu-cluster us-east5-a \
  --dev-tpu-fallback-cluster us-central1 \
  --dev-tpu-name "$USER-gdn" \
  --dev-tpu-type v5p-8 \
  --dev-tpu-allocate-attempts 2 \
  --dev-tpu-allocate-retry-sleep 20 \
  --validation-ray-cluster-auto \
  --validation-ray-cluster-exclude vllm \
  --validation-ray-cluster-exclude big-run \
  --post-check "uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-east5-a --tpu-name $USER-gdn --tests both"
```

When `--hold-dev-tpu` is enabled, `gdnctl` injects a session directive that prefers dev TPU validation/profile commands (`dev-tpu-test` / `dev-tpu-profile`) and allows Ray fallback when needed.
Keep `--post-check` cluster/name aligned to the held allocation (`--cluster` + `--tpu-name`) so checks run on the active dev TPU.
`--validation-ray-cluster-auto` discovers fallback clusters from `infra/marin-*.yaml` and filters them by required TPU type (`--validation-profile-tpu` when profiling is enabled). By default it excludes cluster names containing `vllm` and `big-run`.
With `--resilient`, tests can fall back across modern TPU families (`v5p`/`v5e`/`v6e`) when needed, while profile fallback remains pinned to `--validation-profile-tpu` (for consistent MFU comparisons).

Prompt template used by the loop:
- `scripts/gdn/codex_iteration_prompt.md`
- session directive presets:
  - `training-chunk-kernel-focus` -> `scripts/gdn/session_directives/training-chunk-kernel-focus.md`
  - `control-structure-pivot` -> `scripts/gdn/session_directives/control-structure-pivot.md`
  - `macro-coverage-pivot` -> `scripts/gdn/session_directives/macro-coverage-pivot.md`
  - `associative-summaries` -> `scripts/gdn/session_directives/associative-summaries.md`
  - `xla-first-train-path` -> `scripts/gdn/session_directives/xla-first-train-path.md`

The default prompt is aggressive by design:
- prioritizes train-path control-structure moves over more kernel-local closed-call wins,
- tracks `while` / `conditional` overhead in addition to MFU,
- rejects candidates that worsen control-flow budget unless the end-to-end gain is large,
- prioritizes high-upside kernel redesigns over small tuning,
- disallows standalone scalar-only tweaks,
- requires hotspot-driven escalation when gains are small.
- disallows leaving `Commit: (pending)` in new log entries.
- allows equivalent model reformulations (including removing explicit triangular inversion) if semantics stay correct and performance improves.
- `--dirty-policy stash` restores the stashed tree automatically at the end of each iteration.
- If stash restore conflicts with new iteration edits, the default `--stash-restore-policy warn-keep` keeps the stash and continues; use `--stash-restore-policy fail` for strict behavior.
- `--resilient` is the recommended unattended mode: unlimited failure budget (`--max-failures -1`), retry-enabled codex/post-check paths, and best-effort managed dev TPU hold.
- Validation gate behavior:
  - tests + profile are required each iteration by default (`--validation-mode required`).
  - use `--validation-mode profile-only` for explicit ablation/probe runs where correctness may be intentionally relaxed.
  - execution path prefers held dev TPU and falls back to `ray_run` when dev TPU path fails/unavailable.
  - retries are unbounded by default (`--validation-max-attempts -1`) so TPU queue wait is handled in-loop.
  - for harness debugging only, disable with `--validation-mode off`.
  - pass extra profile env vars with `--validation-profile-env KEY=VALUE` (or `--profile-env KEY=VALUE` on direct `ray-profile` / `dev-tpu-profile` commands).
