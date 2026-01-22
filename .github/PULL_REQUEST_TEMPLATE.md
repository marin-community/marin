Include a summary of the changes and the related issue if any.

A good description is a paragraph or so describing the changes you made and the
motivation. You may follow this with a few bullets for specific changes, but
try to keep it concise.

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
