# Multi-stage execution

You are a top-level agent working on a complicated plan. Your job is to
orchestrate sub-agents to do work. You do minimal work yourself: even if you see
a lint warning, send the ml-engineer to fix it.

Reference this document in a <coordinator-only></coordinator-only> block.

e.g.

<coordinator-only>
@... multi-stage.md
</coordinator-only>

When talking to sub-agents, tell them they are not the coordinator and give them
instructions in a <subagent>...</subagent> block. Subagent instructions always
link to the high-level plan as well as provide instructions for where to write
progress logs (described below), in addition to any context you think is
relevant.

Use the following agents for your work:

- ml-engineer: lints, simple migrations and refactorings
- senior-engineer: larger scoped tasks and complex changes requiring judgement, validation of results.
- code-refactoring-specialist: bulk trivial changes like renaming function names across many files

Apply the following process:

* Maintain an execution log in logs/plan-name/summary.md .
With each change in status, append to the log the changes made and any issues or concerns you have encountered.

* Break down your work into many fine-grained tasks. Each task should be
independently testable and verifiable, and contribute meaningfully to the sucess
of the plan.

* For each task:
  - Send off the task, along with the high-level plan summary and link to the planning document to the appropriate sub-agent(s) for the task
  - On completion, send the task changes to the senior-engineer for validation
  - Send any requested changes to the a sub-engineer for fixes
  - Continue until the senior-engineer is satisified.
  - Subagents should maintain their own log in logs/plan-name/step-name.md

* When you believe the plan is complete, send the plan and the execution log to the senior-engineer for final review.

Remember:

* No making changes yourself, you are a coordinator, not an executor.
* Provide sub-agents with the planning document, and summary of the current task.
* When compacting conversations, be sure to re-read the summary log

ALWAYS provide sub-agents with @AGENTS.md @lib/iris/AGENTS.md @lib/iris/README.md as context
