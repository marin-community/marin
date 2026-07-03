<!-- See .agents/skills/commit/ for full PR guidelines. -->

Lead with what the change does, then why — the problem it fixes or the reason
it's shaped this way. This text becomes the squash-merge commit message, so write
it like one: what a reviewer needs to understand and approve the change, nothing
more. Most PRs are a paragraph or two with no headings. Add markdown (a short
list, a table, a mermaid diagram) only when it makes the change clearer to a
reviewer — not to fill in a template. No "Testing"/"Validation" section.

e.g.

Title: [RL] Fix loss: use global token normalization instead of per-example

"""
This fixes a regression in the DAPO loss computation by switching
from per-example normalization (/ n_i) back to global token
normalization (/ N). Per-example normalization gives shorter responses
disproportionately more gradient weight, which hurts math reasoning
tasks where correct answers often require detailed, longer derivations.
Global normalization weights all examples equally regardless of response
length.
"""

<!-- If this PR addresses an existing issue, include "Fixes #XXXX" below. -->
<!-- For ongoing work, PRs should reference an existing issue. Delete this comment when done. -->
Fixes #
