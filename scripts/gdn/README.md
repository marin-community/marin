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
  --directive-preset triangular-inversion \
  --dirty-policy stash \
  --no-commit-policy count-failure
```

By default, `codex-loop` hides noisy `file update:` / `diff --git` blocks from Codex output. Use `--show-file-updates` to display them.

To allocate and hold a dev TPU for the whole loop session:
```bash
uv run python scripts/gdn/gdnctl.py codex-loop \
  --iterations 5 \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --resilient \
  --hold-dev-tpu \
  --dev-tpu-cluster us-central1 \
  --dev-tpu-fallback-cluster us-east5-a \
  --dev-tpu-name "$USER-gdn" \
  --dev-tpu-type v5p-8 \
  --dev-tpu-allocate-attempts 2 \
  --dev-tpu-allocate-retry-sleep 20 \
  --post-check "uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster us-central1 --tpu-name $USER-gdn --tests both"
```

When `--hold-dev-tpu` is enabled, `gdnctl` injects a session directive forcing dev TPU validation/profile commands (`dev-tpu-test` / `dev-tpu-profile`) and warning if `--post-check` includes `ray-test` or `ray-profile`.
Keep `--post-check` cluster/name aligned to the held allocation (`--cluster` + `--tpu-name`) so checks run on the active dev TPU.

Prompt template used by the loop:
- `scripts/gdn/codex_iteration_prompt.md`
- session directive presets:
  - `triangular-inversion` -> `scripts/gdn/session_directives/triangular-inversion.md`

The default prompt is aggressive by design:
- prioritizes high-upside kernel redesigns over small tuning,
- disallows standalone scalar-only tweaks,
- requires hotspot-driven escalation when gains are small.
- disallows leaving `Commit: (pending)` in new log entries.
- allows equivalent model reformulations (including removing explicit triangular inversion) if semantics stay correct and performance improves.
- `--dirty-policy stash` restores the stashed tree automatically at the end of each iteration.
- If stash restore conflicts with new iteration edits, the default `--stash-restore-policy warn-keep` keeps the stash and continues; use `--stash-restore-policy fail` for strict behavior.
- `--resilient` is the recommended unattended mode: unlimited failure budget (`--max-failures -1`), retry-enabled codex/post-check paths, and best-effort managed dev TPU hold.
