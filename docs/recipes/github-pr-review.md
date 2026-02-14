# Review Prompt

Review this PR. Be extremely terse. No emoji. No fluff.

Format your review as:
1. A 1-2 line description of what the PR does and whether it fulfills its stated objectives.
2. **Specification check** (if a specification exists as a PR comment, linked design doc, or `.agents/projects/` file):
   - Every file in the diff should be traceable to the spec. Flag unexpected files.
   - The implementation should follow the described approach. Flag significant deviations.
   - Described test scenarios should exist and test what they claim.
   - If no specification exists and the PR exceeds ~500 lines of code changes, note that a specification is expected (see `docs/recipes/agent-coding.md`).
3. Tight bullets (if any) for:
   - Bugs or correctness issues
   - Violations of AGENTS.md or coding guidelines
   - Functionality that diverges from the PR description or linked issue

If the PR is clean, say so in one line and stop.

Do NOT comment on:
- Formatting, import order, or linting (handled by pre-commit)
- Style preferences or naming opinions
- Missing docstrings or comments
- Generic best-practice advice

Only flag issues you are confident about. If you are unsure, skip it.
