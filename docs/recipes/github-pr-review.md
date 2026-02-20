# Review Prompt

Review this PR for correctness and behavioral regressions.

Be concise and high-signal. No emoji. No fluff.

Format your review as:
1. A 1-2 line description of what the PR does and whether it fulfills its stated objectives.
2. **Specification check** (if a specification exists as a PR comment, linked design doc, or `.agents/projects/` file):
   - The implementation should follow the described approach. Flag significant deviations.
   - Described test scenarios should exist and test what they claim.
   - If no specification exists and the PR exceeds ~500 lines of code changes, note that a specification is expected (see `docs/recipes/agent-coding.md`).
3. Tight bullets (if any) for:
   - Bugs or correctness issues
   - Violations of AGENTS.md or coding guidelines
   - Functionality that diverges from the PR description or linked issue

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
- Do not include progress narration, process notes, or extra sections.
