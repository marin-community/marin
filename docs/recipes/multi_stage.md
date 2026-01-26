# Multi-stage Deployment

Use this recipe when a task is too large or complex for a single pass. Breaking
work into stages with specialized sub-agents improves quality and reduces errors.

## Why Sub-agents?

LLMs perform poorly with very long prompts, multiple objectives, or large tool
sets. Specialized sub-agents with focused scopes produce better results. Research
shows multi-agent systems outperform single agents by ~90%, though they consume
more tokens—use them when quality matters.

## Agent Selection

Match agent type to task complexity:

| Agent | Use For |
|-------|---------|
| **Code Migration Specialist** | Repetitive refactoring, renames, bulk updates |
| **ML Engineer** | Feature implementation, standard refactoring |
| **Senior Engineer** | Architectural decisions, complex logic, validation |
| **Senior Code Reviewer** | Quality review per AGENTS.md guidelines |

## Orchestration Patterns

**Sequential (default)**: Each stage completes before the next begins. Use when
stages have dependencies or build on previous work.

**Parallel**: Independent stages run concurrently. Use for isolated changes
across different modules.

**Hierarchical**: Break large stages into sub-tasks delegated to lighter agents.
Reserve senior agents for critical decisions.

## Stage Execution Checklist

For **each stage**, complete in order:

- [ ] Provide sub-agent the **full plan** plus stage-specific instructions
- [ ] **Implement**: Assign to appropriate agent (ML Engineer or Senior Engineer)
- [ ] **Verify**: Assign to Senior Engineer to validate against plan
- [ ] **Review**: Assign to Senior Code Reviewer for AGENTS.md compliance
- [ ] **Commit**: Create commit with descriptive message (no self-credit)

Copy this checklist into your plan document.

## Best Practices

- **Start simple**: Begin with 2-3 agents per stage, add complexity only if needed
- **Clear boundaries**: Each agent should have one objective and explicit output format
- **Include context**: Always pass the full plan—agents lack memory across invocations
- **Limit scope**: Keep each stage focused; prefer more small stages over few large ones
- **Fail fast**: If a stage produces poor output, fix the prompt before continuing

## Example Plan Structure

```markdown
## Plan: [Task Name]

### Stage 1: [Name]
- Agent: ML Engineer
- Objective: [Single clear goal]
- Files: [Specific files to modify]
- Output: [Expected deliverable]

### Stage 2: [Name]
...

### Stage Checklist (copy per stage)
- [ ] Implement
- [ ] Verify
- [ ] Review
- [ ] Commit
```
