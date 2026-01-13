# Implementation Recipe Template

This document defines a structured workflow for implementing stages of the
Controller V0 system. Each stage follows this recipe to ensure consistent,
high-quality delivery.

## Per-Stage Execution Template

When implementing a stage from controller-v0.md, follow these phases in order:

---

### Phase 1: Research

**Goal**: Understand the context and existing code before making changes.

Tasks:
- [ ] Read the stage description in `lib/fray/docs/controller-v0.md`
- [ ] Identify all files that will be created or modified
- [ ] Review existing related code:
  - Existing proto definitions in `cluster.proto`
  - Related types in `lib/fluster/src/fluster/cluster/types.py`
  - Any existing controller code in `lib/fluster/src/fluster/cluster/controller/`
- [ ] Check for test patterns in existing tests
- [ ] Note any dependencies on prior stages

**Deliverable**: List of relevant files and understanding of integration points.

---

### Phase 2: Evaluation

**Goal**: Identify gaps between the spec and what's needed for implementation.

Tasks:
- [ ] Compare spec against existing code—what's already done?
- [ ] Identify missing types, imports, or proto messages
- [ ] Flag any ambiguities in the spec that need clarification
- [ ] Determine if the spec's code samples need adjustment for project conventions
- [ ] Verify dependencies are available (proto generated, types exist, etc.)

**Deliverable**: List of gaps, questions, and required adjustments.

---

### Phase 3: Sub-task Breakdown

**Goal**: Create a concrete list of atomic implementation tasks.

Structure sub-tasks as:
1. **Setup**: Create files, add imports, stub out classes/functions
2. **Core Logic**: Implement the main functionality
3. **Tests**: Write unit tests for each component
4. **Integration**: Wire components together if needed

Example sub-task format:
```
- [ ] Create `lib/fluster/src/fluster/cluster/controller/{module}.py`
- [ ] Implement `ClassName` with methods: method1, method2
- [ ] Add unit test `test_{module}.py` covering: scenario1, scenario2
- [ ] Verify tests pass: `uv run pytest tests/test_{module}.py -v`
```

**Deliverable**: Numbered list of sub-tasks with clear acceptance criteria.

---

### Phase 4: Incremental Execution

**Goal**: Implement sub-tasks one at a time, testing after each.

Workflow:
1. Pick the next sub-task
2. Implement the change
3. Run relevant tests immediately:
   ```bash
   uv run pytest tests/test_{module}.py -v
   ```
4. Fix any failures before proceeding
5. Repeat until all sub-tasks complete

Rules:
- Never proceed with failing tests
- Commit logical chunks if the stage is large
- Keep changes minimal—don't refactor unrelated code

**Deliverable**: Working implementation with all tests passing.

---

### Phase 5: Validation

**Goal**: Verify the stage is complete and correct.

Tasks:
- [ ] All new tests pass
- [ ] Existing tests still pass: `uv run pytest lib/fluster/tests/ -v`
- [ ] Type checking passes: `uv run mypy lib/fluster/src/`
- [ ] Linting passes: `uv run python infra/pre-commit.py --all-files`
- [ ] Manual review: does the implementation match the spec?
- [ ] Integration check: can this stage be used by subsequent stages?

**Deliverable**: Clean test run, passing type checks, lint-clean code.

---

### Phase 6: Code Review

**Goal**: Get feedback from senior-code-reviewer agent before finalizing.

Provide to reviewer:
- Summary of changes made
- List of files modified/created
- Key design decisions and rationale
- Any deviations from the spec

Address all feedback before proceeding.

**Deliverable**: Reviewer-approved changes.

---

### Phase 7: Commit

**Goal**: Create a clean commit with descriptive message.

Commit message format:
```
[fluster] Implement {Stage Name}

- {Brief description of what was added}
- {Key components implemented}
- {Tests added}

Part of controller-v0 implementation.
```

Final checks:
- [ ] `git status` shows only relevant changes
- [ ] No debug code or temporary files included
- [ ] Commit message accurately describes changes

---

## Stage Checklist Summary

For quick reference, each stage must complete:

| Phase | Key Output |
|-------|-----------|
| 1. Research | Understanding of context and dependencies |
| 2. Evaluation | List of gaps and adjustments needed |
| 3. Sub-tasks | Concrete implementation plan |
| 4. Execution | Working code with passing tests |
| 5. Validation | All checks green |
| 6. Review | Feedback addressed |
| 7. Commit | Clean, descriptive commit |

---

## Controller V0 Stages

Reference: `lib/fray/docs/controller-v0.md`

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Proto Updates for Controller-Worker Communication | ✓ Completed |
| 2 | Controller Core Data Structures | Pending |
| 3 | Worker Registry (Static Configuration) | Pending |
| 4 | Job Scheduler Thread | Pending |
| 5 | Worker Heartbeat Monitor | Pending |
| 6 | Job Failure and Retry Logic | Pending |
| 7 | Controller Service Implementation | Pending |
| 8 | Integration Test - End to End | Pending |
| 9 | Dashboard (HTTP) | Pending |

---

## Agent Handoff Instructions

When delegating a stage to the `ml-engineer` agent, provide:

```
## Stage {N}: {Stage Name}

### Context
- Spec location: lib/fray/docs/controller-v0.md (Stage {N} section)
- Recipe: lib/fluster/docs/impl-recipe.md
- Target directory: lib/fluster/src/fluster/cluster/controller/

### Your Task
Follow the implementation recipe phases 1-7 to complete Stage {N}.

### Key Requirements from Spec
{Copy relevant section from controller-v0.md}

### Dependencies
- Prior stages: {list any prerequisites}
- Proto messages: {list any required proto types}

### Acceptance Criteria
- All tests in spec pass
- Type checking passes
- Code review approved
- Clean commit created
```
