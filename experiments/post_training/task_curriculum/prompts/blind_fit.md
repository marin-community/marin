# Blind-fit judgment prompt

Use a separate `gpt-5.6-sol` high-reasoning context. Attach the candidate curriculum and only the blind task IDs and
instructions. Do not attach generation metadata, holistic reviews, prior fit judgments, or answers.

```text
Judge whether each independently generated task has a useful home in this subject curriculum. Inspect the complete
task, not its nouns or output format. Groups are never task targets.

For each task:
- `exact`: one capability covers the complete task and its decisive operation.
- `ambiguous`: two or more capabilities each plausibly cover the complete task; list all acceptable capabilities.
- `gap`: no capability covers the complete task. Partial coverage by several capabilities is still a gap.
- `invalid`: the task is malformed, materially underdetermined, or not genuinely in the subject.

Do not reward a capability merely because its wording resembles the task. Do not penalize benign changes in data,
surface form, or sampling-facet value. Name the decisive operation and explain the result briefly. Then return the
strict blind-fit JSON contract, verify that every input task appears exactly once, compute the four counts, set the
numerator to exact plus defensible ambiguous, set the denominator to total minus invalid, and identify repeated gaps
that indicate a missing operation family.
```
