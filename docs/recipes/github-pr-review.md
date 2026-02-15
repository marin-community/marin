# Review Prompt

Review this PR for correctness and behavioral regressions.

Be concise and high-signal. No emoji. No fluff.

Execution constraints:
- Use at most 2 analysis passes:
  1) Diff-first scan for risk areas
  2) Targeted file checks only where risk is detected
- Prefer a PR diff tool first (for example, `gh pr diff` when available); only open full files when necessary to confirm a concrete issue.
- If no high-confidence issues are found after these passes, stop.

Scope:
- Only report high-confidence findings that are likely to cause:
  - Bugs or correctness issues
  - Violations of AGENTS.md or coding guidelines
  - Functionality divergence from the PR description or linked issue
- If a specification exists (issue description, design doc, acceptance criteria, or inline requirements), verify the code adheres to it and flag concrete mismatches.
- Ignore formatting, import order, lint/style preferences, naming opinions, missing docstrings/comments, and generic best-practice advice.

Output contract:
- Return exactly one final review.
- Keep output compact and high-signal.
- Use this format only:

1. `<1-2 lines: what the PR changes and whether it meets objectives>`
2. Findings:
- `<issue 1 or "None">`
- `<issue 2 if needed>`
- `<issue 3 if needed>`

Rules:
- Max 3 findings.
- Each finding must be one line and include file path (and line if known).
- If clean: write `- None` and stop.
- Do not include progress narration, process notes, or extra sections.
