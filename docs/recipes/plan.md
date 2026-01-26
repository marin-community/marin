# Plan Template

Use this template when constructing an implementation plan. All sections are
**required**. If a section is not relevant, include it with "N/A" and a brief
justification.

---

## Goal

One paragraph describing what this plan achieves. State:

- **What**: The specific outcome or deliverable
- **Why**: The motivation or problem being solved
- **Scope boundaries**: What is explicitly included and excluded

Example:
> This plan implements a CLI for the Iris autoscaler, enabling operators to
> inspect cluster state, trigger manual scaling actions, and view scaling
> history. It does not include changes to the autoscaling logic itself or
> modifications to the web dashboard.

---

## Non-Goals

Explicitly list what this plan will **not** address. These are reasonable
objectives that are deliberately excluded to prevent scope creep.

- [Non-goal 1]: Brief rationale
- [Non-goal 2]: Brief rationale

Example:
> - **Backward compatibility with v1 API**: The v1 API is deprecated and has no
>   active users
> - **Windows support**: All production deployments run on Linux

---

## Design Overview

High-level description of the approach. Include:

1. **Architecture**: How components fit together
2. **Data flow**: How information moves through the system
3. **Key decisions**: Important choices made and why

Use diagrams where helpful, but always accompany them with written explanation.

Example:
> The CLI communicates with the autoscaler via its existing gRPC API. No new
> endpoints are required. The CLI is stateless—all state lives in the
> autoscaler service. We use Click for argument parsing because it's already a
> project dependency and provides automatic help generation.

---


## Files Modified or Created

**This section is mandatory.** Provide a directory tree showing all files
affected by this plan, annotated with their status.

### Directory Tree

```
lib/module/src/module/
  cli.py <new>
  platform.py <new>
  logging.py <modified>
  agents.py <modified>
  tests/
    test_cli.py <new>
    test_agents.py <modified>
```

Annotations:
- `<new>` - File will be created
- `<modified>` - Existing file will be changed
- `<deleted>` - File will be removed
- `<renamed: old_name.py>` - File will be renamed

### File-by-File Overview

**This section is mandatory.** For each file listed above, describe:

1. **Purpose**: What role does this file play?
2. **Changes**: What specific modifications will be made?
3. **Key interfaces**: Any new functions, classes, or APIs introduced

#### `cli.py` (new)

Create the Click-based CLI entry point. Implements:

- `list` command: Display all managed clusters with status
- `scale` command: Trigger manual scaling with `--replicas` flag
- `history` command: Show recent scaling events

Key interfaces:
```python
@click.group()
def cli() -> None: ...

@cli.command()
@click.option("--cluster", required=True)
def list(cluster: str) -> None: ...
```

#### `platform.py` (new)

Cross-platform filesystem utilities for config and cache paths. Implements
`get_config_dir()` and `get_cache_dir()` following XDG conventions on Linux
and appropriate conventions on macOS.

#### `agents.py` (modified)

Add the `query()` method to the `Agent` class to support CLI inspection.
Modify `AgentStatus` enum to include `SCALING` state.

#### `logging.py` (modified)

Change default log level from DEBUG to INFO. Add structured JSON output option
for CLI consumption via `--json` flag.

---

## Alternatives Considered

Document other approaches that were evaluated. For each alternative:

1. Brief description
2. Trade-offs (pros and cons)
3. Why it was not chosen

### Alternative A: [Name]

[Description]

**Trade-offs**: [Why not chosen—focus on the decisive factors]

### Alternative B: [Name]

[Description]

**Trade-offs**: [Why not chosen]

---

## Implementation Steps

Ordered list of concrete steps to complete this plan. Each step should be:

- **Atomic**: Can be completed and verified independently
- **Testable**: Has clear success criteria
- **Sized appropriately**: Neither too granular nor too broad
- **Narrow, not wide**: Prefer steps which thread some end-to-end behavior vs steps and help validate the idea.

### Step 1: [Description]

**Files**: `cli.py`, `test_cli.py`

**Actions**:
1. Create `cli.py` with Click group and `list` command stub
2. Add unit tests for argument parsing
3. Verify: `uv run python -m module.cli list --help` shows usage

### Step 2: [Description]

**Files**: `agents.py`, `test_agents.py`

**Actions**:
1. Add `query()` method to `Agent` class
2. Update existing tests for new method
3. Verify: All tests pass with `uv run pytest tests/test_agents.py`

---

## Cross-Cutting Concerns

Address these areas as relevant:

### Testing Strategy

- Unit tests for [components]
- Integration tests for [workflows]
- Manual verification steps

### Error Handling

- How errors are surfaced to users
- Retry/recovery behavior
- Logging and diagnostics

### Security

- Authentication/authorization changes
- Input validation
- Sensitive data handling

### Performance

- Expected load/scale
- Potential bottlenecks
- Benchmarking approach

### Observability

- Metrics added
- Log formats
- Tracing integration

---

## Dependencies

External factors that must be in place before or during implementation:

| Dependency | Status | Owner | Notes |
|------------|--------|-------|-------|
| gRPC API v2 deployed | Complete | @alice | Deployed 2026-01-20 |
| Click library >=8.0 | Complete | - | Already in pyproject.toml |
| Access to staging cluster | Pending | @bob | Needed for integration tests |

---

## Open Questions

Issues requiring resolution before or during implementation. For each:

- State the question clearly
- Provide context
- Suggest who should weigh in

1. **[Question]**: [Context and options being considered] — @person
2. **[Question]**: [Context] — @person

---

## Plan Review

* Now that you have constructed the plan, what issues do you observe in it?
* Does it adhere to your @AGENTS.md guidelines?

Write out potential problems with your plan here and suggestions for improvements.
