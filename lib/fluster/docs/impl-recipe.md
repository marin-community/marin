# Implementation Recipe Template

This document defines a structured workflow for implementing a feature or
component in the fluster-zero system. Each feature follows this recipe to ensure
consistent, high-quality delivery.

### Phase 1: Research

**Goal**: Understand the context and existing code before making changes.
**Deliverable**: List of relevant files and understanding of integration points.

Tasks:
- [ ] Read the feature description from the implementation doc provided
- [ ] Use the explore agent to find and review related code or protocol files
- [ ] Check for test patterns in existing tests
- [ ] Note any dependencies on other features

---

### Phase 2: Evaluation

**Goal**: Identify gaps between the spec and what's needed for implementation.
**Deliverable**: List of gaps, questions, and required adjustments.

Tasks:
- [ ] Compare spec against existing code—what's already done?
- [ ] Identify missing types, imports, or proto messages
- [ ] Flag any ambiguities in the spec that need clarification
- [ ] Determine if the spec's code samples need adjustment for project conventions
- [ ] Verify dependencies are available (proto generated, types exist, etc.)

---

### Phase 3: Sub-task Breakdown
**Goal**: Create a concrete list of atomic implementation tasks.
**Deliverable**: Numbered list of sub-tasks with clear acceptance criteria.

Sub tasks should be independently testable or verifiable.

Prefer "spiral" sub-tasks over "linear" ones: changes that touch multiple files but test a single concept, which is then expanded.

Example sub-tasks for a new RPC method:

* Add new proto message and stub implementation in the server, write a test which wires up client and server
* Implement initial logic in the server, update test
* Add client-level functionality e.g. tracing, update tests
* etc

---

### Phase 4: Incremental Execution

**Goal**: Implement sub-tasks one at a time, testing after each.

Workflow:
1. Pick the next sub-task
2. Implement the change
3. Run relevant tests immediately
4. Fix any failures before proceeding
5. Repeat until all sub-tasks complete

Rules:
- Never proceed with failing tests
- Keep changes minimal—don't refactor unrelated code

**Deliverable**: Working implementation with all tests passing.

---

### Phase 5: Validation

**Goal**: Verify the stage is complete and correct.

Tasks:
- [ ] All new tests pass
- [ ] Existing tests still pass
- [ ] Type checking passes: `uv run pyrefly`
- [ ] Linting passes: `uv run python infra/pre-commit.py --all-files`

**Deliverable**: Clean test run, passing type checks, lint-clean code.

---

### Phase 6: Code Review

**Goal**: Get feedback from senior-code-reviewer agent before finalizing.

Provide to reviewer:
- The feature and source implementation document
- Summary of changes made
- List of files modified/created
- Key design decisions and rationale
- Any deviations from the spec

Address all feedback before proceeding.

**Deliverable**: Reviewer-approved changes.

Tasks:

- [ ] Reviewer signed off on the changes

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

YOU MUST COMMIT WITH GIT BEFORE CONSIDERING THIS PHASE COMPLETE
