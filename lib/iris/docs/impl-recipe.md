# Implementation Recipe Template

This recipe provides a structured workflow for implementing individual stages of a larger design. Each stage should be completed as a focused, reviewable unit of work.

## Purpose

When implementing a multi-stage design, break the work into discrete phases. This recipe guides you through completing a single phase: from verifying assumptions through to a committed, reviewed change.

## Prerequisites

- Read the top-level design document at `docs/iris-coscheduling.md`
- Identify which stage you are implementing
- Ensure you have access to the codebase and can run tests

## Workflow Steps

### 1. Explore & Verify

Before writing any code, verify that the design assumptions for this stage are still valid.

- Read the relevant section of the design document
- Explore the current codebase to understand existing structures
- Verify that dependencies from previous stages are in place
- Check that the interfaces you plan to use exist and behave as expected
- If assumptions are invalid, stop and update the design document first

```bash
# Example: explore the current state of the module
uv run python -c "from iris import cluster; print(dir(cluster))"
```

### 2. Define Tasks

Create a concrete todo list of implementation tasks for this stage.

- Break the stage into small, testable units of work
- Each task should be completable in isolation
- Order tasks by dependency (implement foundations first)
- Include tasks for tests, not just implementation

Example task breakdown:
```
- [ ] Add new field to ResourceSpec proto
- [ ] Update Python bindings for ResourceSpec
- [ ] Add validation logic for new field
- [ ] Write tests for validation
- [ ] Update existing call sites
```

### 3. Implement

Execute each task, writing code according to the design.

- Follow project coding conventions (see `AGENTS.md`)
- Write tests alongside implementation, not after
- Keep changes minimal and focused on the current stage
- Do not introduce unrelated refactors
- Run tests frequently to catch regressions early

```bash
# Run tests for the module you're modifying
uv run pytest lib/iris/tests/ -v
```

### 4. Senior Code Review

Before committing, send your changes for review.

- Use the senior-code-reviewer agent to review pending changes
- Provide context about what stage you're implementing
- Be prepared to explain design decisions

### 5. Fix Review Issues

Address any issues raised during code review.

- Make requested changes
- Re-run tests after each fix
- If you disagree with feedback, discuss before proceeding
- Request re-review if substantial changes were made

### 6. Commit

Once review is complete and all issues are addressed, commit the changes.

- Use the `/commit` skill to create a well-formatted commit
- Ensure the commit message references the stage being implemented
- Do not include unrelated changes in the commit

---

## Stage-Specific Instructions

<!-- Fill in this section with details for the specific stage being implemented -->

### Stage: [STAGE NAME]

**Objective**: [Brief description of what this stage accomplishes]

**Files to Modify**:
- `path/to/file1.py` - [what changes]
- `path/to/file2.py` - [what changes]

**New Files to Create**:
- `path/to/new_file.py` - [purpose]

**Code Snippets**:

```python
# Example implementation pattern for this stage
def example_function():
    pass
```

**Verification Commands**:

```bash
# Commands to verify the implementation is correct
uv run pytest lib/iris/tests/test_specific.py -v
```

**Dependencies**:
- [List any stages that must be completed first]

**Acceptance Criteria**:
- [Specific criteria that indicate this stage is complete]

---

## Quality Checklist

Before committing, verify all items:

- [ ] All tests pass (`uv run pytest lib/iris/tests/ -v`)
- [ ] No regressions introduced in existing functionality
- [ ] Code follows project conventions (see `AGENTS.md`)
- [ ] Changes are minimal and focused on this stage only
- [ ] New code has appropriate test coverage
- [ ] No unrelated refactors or cleanups included
- [ ] Commit message clearly describes the change

## Troubleshooting

**Tests fail after implementation**:
- Check if you're missing a dependency from a previous stage
- Verify your changes don't break existing interfaces
- Run tests in isolation to identify the specific failure

**Design assumptions are wrong**:
- Stop implementation
- Document what is different from the design
- Update the design document before proceeding
- Consider whether this affects other stages

**Review feedback requires major changes**:
- Re-evaluate whether the design needs updating
- Make changes incrementally, testing after each
- Request re-review once substantial changes are complete
