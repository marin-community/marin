# Independent subject-local progression review

Use one fresh `gpt-5.6-sol` high-reasoning reviewer for the subject. Attach the same compact subject packet, the
proposal, and `prompts/learning_progression.md`; do not attach another subject, the proposal transcript, or an earlier
review. Bind structured output to `LearningProgressionReview`.

```text
Judge every proposed edge against the learning-enablement rule. Accept an edge only when both witness families use the
prerequisite capability's stated outcome, reuse the same foundation, and add one shared main operation in the
dependent capability. Reject course order, domain relabeling, shared vocabulary, general sophistication, artifact
handoff, narrow scopes, cycles, and redundant transfers already represented by a closer path.

Then scan the complete subject packet for clear missing immediate edges. Supply each omission as a complete compact
edge with two witness-family sketches. Do not construct an exhaustive course DAG and do not rewrite a rejected pair.
Return one verdict for every proposed edge exactly once. Recommend `revise` when any edge is rejected or an omission
is supplied; otherwise recommend `accept`.
```
