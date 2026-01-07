# Recipe: Writing Implementation Plans

## Purpose

An **implementation plan** transforms a design document into a concrete, staged roadmap for building the feature. While a design doc explains *what* and *why*, an implementation plan specifies *how* and *in what order*, with each stage independently testable.

**Relationship to design docs**: Design docs should remain tight 1-pagers (~200-400 lines). Implementation plans are the next step—expanding the "Implementation Outline" section into a full staged plan with validation criteria.

---

## When to Write an Implementation Plan

Write an implementation plan when:
- The design involves changes to multiple subsystems
- The implementation spans more than one logical phase
- You need coordination between agents or team members
- The work requires incremental validation (not just "run tests at the end")

Skip formal implementation plans for:
- Single-file bug fixes
- Straightforward feature additions with obvious implementation
- Changes covered by an existing recipe

---

## Implementation Plan Structure

### 1. Background Section

The background section establishes context by documenting research into the codebase. It should include:

**Source Document Reference**
- Link to the design doc this plan implements
- Summary of the design goals (1-2 sentences)

**Affected Code**
- List files and modules that will be modified
- Include line references for key functions/classes
- Group by subsystem when helpful

**Dependencies**
- External libraries required
- Internal modules that must be understood
- Related systems that may be affected

Example:
```markdown
## Background

**Design**: [Fray Design Specification](.agents/docs/fray_design_spec.md)

**Goal**: Replace Ray task dispatch with Fray's storage-first execution model.

### Affected Code

**Task Dispatch**
- `lib/fray/src/fray/context.py:42` - FrayContext.run()
- `lib/fray/src/fray/cluster.py:118` - LocalCluster task submission

**Zephyr Integration**
- `lib/zephyr/src/zephyr/backends/ray.py:55` - RayBackend.execute()
- `lib/zephyr/src/zephyr/backends/base.py:22` - Backend protocol

**Tests**
- `tests/fray/test_context.py` - Context unit tests
- `tests/zephyr/test_backends.py` - Backend integration tests
```

### 2. Staged Implementation

Each stage must be:
- **Logically disjoint**: Completable without starting other stages
- **Independently testable**: Has clear validation criteria
- **Atomic**: Can be committed and reviewed separately

#### Stage Structure

```markdown
## Stage N: [Descriptive Title]

### Objective
[1-2 sentences describing what this stage accomplishes]

### Changes
- [ ] Change 1 with file:line reference
- [ ] Change 2 with file:line reference

### Validation
- [ ] Test: `uv run pytest tests/path/test_file.py::test_name -v`
- [ ] Manual: [Description of manual verification if needed]

### Completion Criteria
Stage is complete when all validation checks pass.
```

#### Stage Types

**Code Stages**: Implement functionality
```markdown
## Stage 1: Implement FrayContext.run()

### Objective
Add basic task dispatch capability to FrayContext.

### Changes
- [ ] Add `run(f, *args, **kwargs) -> FrayFuture` method to FrayContext
- [ ] Implement FrayFuture with status(), wait(), result() methods
- [ ] Add TaskStatus enum (PENDING, IN_PROGRESS, FINISHED, FAILED)

### Validation
- [ ] `uv run pytest tests/fray/test_context.py::test_run_simple_task -v`
- [ ] `uv run pytest tests/fray/test_context.py::test_run_with_failure -v`
```

**Research Stages**: Gather information, produce artifacts
```markdown
## Stage 0: API Research

### Objective
Document current Ray API usage patterns across the codebase.

### Changes
- [ ] Search for all `@ray.remote` decorators
- [ ] Categorize by pattern (task, actor, resource requirements)
- [ ] Document findings in `.agents/docs/ray-usage-audit.md`

### Validation
- [ ] Sub-agent reviews audit document for completeness
- [ ] Document covers all files in `lib/marin/src/marin/`
```

**Migration Stages**: Update existing code
```markdown
## Stage 3: Migrate Zephyr Backend

### Objective
Replace RayBackend with FrayBackend in Zephyr.

### Changes
- [ ] Create FrayBackend implementing Backend protocol
- [ ] Update `create_backend()` to support "fray" backend type
- [ ] Add deprecation warning to RayBackend (if requested)

### Validation
- [ ] `uv run pytest tests/zephyr/test_backends.py -v`
- [ ] `uv run pytest tests/zephyr/test_integration.py -k fray -v`
- [ ] Existing tests pass with both ray and fray backends
```

### 3. Dependencies Between Stages

When stages have dependencies, make them explicit:

```markdown
## Stage Dependencies

```
Stage 0 (Research) ─┐
                    ├──► Stage 2 (Integration)
Stage 1 (Core API) ─┘
                          │
                          ▼
                    Stage 3 (Migration)
```

Stages 0 and 1 can proceed in parallel.
Stage 2 requires both Stage 0 and Stage 1.
Stage 3 requires Stage 2.
```

### 4. Test-Driven Development

Each code stage should follow TDD:

1. **Write failing test first** - Captures expected behavior
2. **Implement minimum code** - Make the test pass
3. **Refactor** - Clean up while tests stay green

Example stage with TDD workflow:
```markdown
## Stage 1: Implement TaskStatus tracking

### TDD Workflow

**Step 1: Write test**
```python
# tests/fray/test_context.py
def test_task_status_lifecycle():
    ctx = FrayContext.create("local://")
    future = ctx.run(lambda: 42)

    assert future.status() in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
    future.wait()
    assert future.status() == TaskStatus.FINISHED
    assert future.result() == 42
```

**Step 2: Run test (expect failure)**
```sh
uv run pytest tests/fray/test_context.py::test_task_status_lifecycle -v
# Expected: ImportError or AttributeError
```

**Step 3: Implement**
- Add TaskStatus enum
- Add status tracking to FrayFuture

**Step 4: Verify**
```sh
uv run pytest tests/fray/test_context.py::test_task_status_lifecycle -v
# Expected: PASSED
```
```

---

## Process for Creating an Implementation Plan

### Step 1: Research the Codebase

Before writing the plan:

1. **Read the design doc thoroughly** - Understand goals and non-goals
2. **Search for affected code** - Use grep/glob to find relevant files
3. **Trace call paths** - Understand how components interact
4. **Identify test coverage** - Find existing tests to extend

Use the Task tool with `subagent_type=Explore` for broad codebase exploration.

### Step 2: Write the Background Section

Document your research findings:
- List all files that will be touched
- Note key functions/classes with line numbers
- Identify dependencies and potential conflicts

### Step 3: Decompose into Stages

Break the work into stages by asking:
- What can be tested independently?
- What must happen before other work can begin?
- Where are the natural boundaries between subsystems?

**Good decomposition signs:**
- Each stage has 2-5 concrete changes
- Validation is straightforward (usually 1-3 test commands)
- Stages can be reviewed in isolation

**Bad decomposition signs:**
- Stage has 10+ changes
- Validation requires running the entire system
- Later stages frequently need to modify earlier work

### Step 4: Define Validation Criteria

For each stage, specify:
- Exact test commands to run
- Expected outcomes
- Manual verification steps if needed

### Step 5: Review and Refine

Before executing:
- Verify stages are truly independent
- Ensure validation criteria are concrete
- Check that the plan covers all design requirements

---

## Example Implementation Plan

See `.agents/docs/fray-migration.md` and `.agents/docs/zephyr-migration.md` for real examples of migration plans.

Here's a minimal example:

```markdown
# Implementation Plan: Add Metrics Export to FrayContext

## Background

**Design**: [metrics-export-design.md]

**Goal**: Export task execution metrics (duration, status, retries) in Prometheus format.

### Affected Code
- `lib/fray/src/fray/context.py:42` - FrayContext class
- `lib/fray/src/fray/metrics.py` - (new file)
- `tests/fray/test_metrics.py` - (new file)

### Dependencies
- `prometheus_client` library (add to pyproject.toml)

## Stage 1: Add Metrics Collection

### Objective
Instrument FrayContext to collect basic task metrics.

### Changes
- [ ] Create `lib/fray/src/fray/metrics.py` with MetricsCollector class
- [ ] Add task_duration_seconds histogram
- [ ] Add task_status_total counter
- [ ] Integrate MetricsCollector into FrayContext.run()

### Validation
- [ ] `uv run pytest tests/fray/test_metrics.py::test_metrics_collected -v`
- [ ] `uv run pytest tests/fray/test_metrics.py::test_duration_histogram -v`

## Stage 2: Add HTTP Export Endpoint

### Objective
Expose metrics via HTTP for Prometheus scraping.

### Changes
- [ ] Add `start_metrics_server(port)` function
- [ ] Integrate with FrayController lifecycle
- [ ] Add configuration option for metrics port

### Validation
- [ ] `uv run pytest tests/fray/test_metrics.py::test_http_endpoint -v`
- [ ] Manual: `curl localhost:9090/metrics` returns Prometheus format

## Stage Dependencies

Stage 1 ──► Stage 2

Stage 2 requires Stage 1's MetricsCollector.
```

---

## Guidelines for Agents

When creating an implementation plan:

1. **Research first** - Spend time understanding the codebase before planning
2. **Be specific** - Include file paths, line numbers, test commands
3. **Keep stages small** - 2-5 changes per stage is ideal
4. **Validate incrementally** - Each stage must have runnable tests
5. **Document dependencies** - Make stage ordering explicit
6. **Use TDD** - Write tests before implementation code
7. **Update as you go** - Mark checkboxes complete, add notes about issues

When executing an implementation plan:

1. **Work one stage at a time** - Don't skip ahead
2. **Run validation after each change** - Catch issues early
3. **Commit at stage boundaries** - Keep changes reviewable
4. **Update the plan** - Note any deviations or discoveries

---

## Tasks Checklist

For agents creating an implementation plan:

- [ ] Read and understand the design document
- [ ] Research affected code in the codebase
- [ ] Write background section with file references
- [ ] Decompose into logically disjoint stages
- [ ] Define validation criteria for each stage
- [ ] Document stage dependencies
- [ ] Review plan for completeness

For agents executing an implementation plan:

- [ ] Read the full plan before starting
- [ ] Execute stages in dependency order
- [ ] Run validation after each stage
- [ ] Mark stages complete as you go
- [ ] Commit changes at stage boundaries
- [ ] Report blockers immediately

---

## See Also

- `docs/recipes/design_doc.md` - How to write design documents
- `docs/recipes/fix_issue.md` - Workflow for fixing issues
- `.agents/docs/fray-migration.md` - Example migration plan
- `.agents/docs/zephyr-migration.md` - Example migration patterns
- `AGENTS.md` - General coding guidelines
