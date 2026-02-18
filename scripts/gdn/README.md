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
  --directive-preset triangular-inversion \
  --dirty-policy stash \
  --no-commit-policy count-failure
```

Prompt template used by the loop:
- `scripts/gdn/codex_iteration_prompt.md`

The default prompt is aggressive by design:
- prioritizes high-upside kernel redesigns over small tuning,
- disallows standalone scalar-only tweaks,
- requires hotspot-driven escalation when gains are small.
- disallows leaving `Commit: (pending)` in new log entries.
