# TaskCompendium

Read root AGENTS.md and TESTING.md. This is an independent Python package with a
separate uv workspace, so it can be installed into Harbor without Marin's JAX stack.

Use `uv run --project lib/taskcompendium --group test pytest lib/taskcompendium/tests`
for safe package tests. Harbor tests require `--extra harbor`. Preserve the default
marker exclusions. Use the repository `infra/pre-commit.py` entry point for lint.

Reuse the pinned tasktrove-verify ontology and graders. Submission extraction and
agent resource visibility belong here; source answer recovery does not. Never
run imported scripts, tests, or submitted programs on the host verifier runtime.

## Agent-facing task instructions

Write the task request directly. `TaskSpecification.instructions`, rendered prompts,
and public task resources must not disclose evaluation machinery: judges, judgers,
verifiers, graders, rewards, evaluation scores, hidden tests, or reference answers.
Keep these in verifier/oracle metadata and resources, never in agent-facing text.

Preserve actual requirements when removing source evaluation commentary. Use
"Write your answer to `/app/answer.txt`" instead of describing what checks that file.
Use "Fix the bug while preserving existing behavior" instead of explaining that
pytest will grade the solution. Required formats, file paths, observable behavior,
and user-requested tests remain part of the task.

Importers should rewrite known source boilerplate explicitly, without deleting
unrelated domain language such as credit scores or requests to verify reproducibility.
Prompt generation and lowering must not add evaluation commentary back. Review both
canonical instructions and every rendered variant before publishing; add regressions
for source templates that leak evaluation details. Keep source archives unchanged.
