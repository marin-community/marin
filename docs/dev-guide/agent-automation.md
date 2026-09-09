# Agent automation model and cost audit

This inventory covers active repository code and workflow configuration that
launches an LLM. Historical project notes and logbooks can contain old command
examples; they do not launch agents.

Every launch must select a model and effort explicitly. Cost-sensitive review
should use a budget model. The concrete model can differ by vendor: Codex lint
and local code review use `gpt-5.6-terra` at low effort, while the checked-in
Claude lint default uses `claude-haiku-4-5-20251001` at low effort.

## Instruction context reductions

| Instruction file | Baseline lines | Current lines | Retained | Moved or routed |
| --- | ---: | ---: | --- | --- |
| `AGENTS.md` | 249 | 208 | Repository invariants, safety, validation entry points, and workflow routing | Echo syntax to `consult-echo`; PR procedure to `commit` and `writing-style` |
| `lib/iris/AGENTS.md` | 174 | 116 | Iris data, RPC, time, persistence, environment, and task-setup invariants | Source and backend reference to `docs/architecture.md` |
| `lib/zephyr/AGENTS.md` | 60 | 45 | Blocking RPC, data-flow, failure, and backend invariants | Source map to `README.md` |

Skills no longer request another read of root `AGENTS.md`. They still route to
module instructions and testing documents when those add scoped requirements.

| Launcher | Owner path | Authentication | Model and effort | Maximum calls per run |
| --- | --- | --- | --- | ---: |
| `infra/linter.py` default | `infra/lint/` | Local Claude subscription; `ANTHROPIC_API_KEY` is removed | `claude-haiku-4-5-20251001`, low | One per selected lane; optional composer adds one |
| `lint-review` Codex example | `.agents/skills/lint-review/` | Local Codex CLI auth; parent session and Loom markers are removed | `gpt-5.6-terra`, low | One per selected lane; optional composer adds one |
| `infra/cron/nightshift_cleanup.py` | `infra/cron/` | Inherited local Claude CLI auth | `claude-opus-4-8`, high | Four scouts and one merge agent |
| `infra/cron/nightshift_ci_tests.py` | `infra/cron/` | Inherited local Claude CLI auth | `claude-opus-4-8`, high | One |
| `infra/cron/nightshift_doc_drift.py` | `infra/cron/` | Inherited local Claude CLI auth | `claude-opus-4-8`, high | One |
| `.github/workflows/ops-claude.yaml` | `.github/workflows/` | `CLAUDE_CODE_OAUTH_TOKEN` or `CLAUDE_MAX_OAUTH_TOKEN` | Opus/high for implementation, Sonnet/low for triage and autofix | One per triggered job |
| `.github/workflows/ops-agent-prose-cleanup.yaml` | `.github/workflows/` | `CLAUDE_CODE_OAUTH_TOKEN` or `CLAUDE_MAX_OAUTH_TOKEN` | `claude-opus-4-8`, high | One; workflow is disabled |
| `infra/loom/Pulumi.marin-loom.yaml` profiles | `infra/loom/` | Loom-managed agent credentials | Explicit Codex model and effort per profile | One primary session; profile concurrency bounds independent sessions |

`scripts/ci/claude_runner.py` is the shared process wrapper for the three
nightshift workflows. It rejects a missing model, a moving model alias, or a
missing effort tier. `infra/linter.py` rejects commands that omit either model
or effort.

Re-run this audit with targeted searches:

```bash
rg -n --hidden --glob '!**/*.lock' --glob '!uv.lock' \
  '(codex (exec|review)|claude (-p|--print)|run_claude\(|agent_command|claude_args:|sessions\.launch)' \
  .agents .github infra scripts lib
```

For each new result, record the authentication source, exact model, effort,
fan-out bound, and subsystem path in this table. Do not copy historical command
examples into active guidance unless they satisfy the current policy.
