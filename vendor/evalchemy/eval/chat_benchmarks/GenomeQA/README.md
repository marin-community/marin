# GenomeQA

This adapter evaluates the original 5,200 questions from
[GenomeQA](https://arxiv.org/abs/2604.05774). Marin issue
[#10981](https://github.com/marin-community/marin/issues/10981) records the
protocol choices and alternatives. Registry availability does not admit the
benchmark into Marin's official evaluation policy.

`GenomeQA` loads the pinned HF `binary` and `mcq` files.
It does not run tools, retrieve context, shuffle options, or
optimize prompts. Question identities include format, subtask, and original ID.

The source verifies file checksums and preserves question text and option order.
It reads `SIMPLE_SYSTEM_PROMPT` as a literal from the pinned original runner;
it never executes that Python file. The runner prompt is not bundled here.
Figure 8 of the CC BY paper is shorter and is not an interchangeable prompt.
The system-prompt hash and source revisions are recorded in result metadata.

Generation uses one response, temperature 0.7, top-p 0.95, and a 16,384-token
cap. Conflicting settings fail before generation. Full-prompt context preflight
requires the served tokenizer and context length; inputs are never truncated or
filtered to fit. Provider reasoning-budget semantics and effective wire settings
must still be checked before a live evaluation. Thinking effort is model-specific.

The grader reuses shared final-content selection and thinking-marker cleanup,
then applies a GenomeQA-specific letter parser to the last `### Answer` section.
Without that heading it requires an explicit `Answer: X`. Conflicting or invalid
letters fail. Empty answers, reported provider refusals, and reported output-limit
truncations receive zero; transport failures remain ungraded. Refusal and
truncation detection require structured response metadata (`CompletionText`);
plain-text responses cannot establish either condition. Verify that each backend
preserves this metadata before claiming those protocol guarantees. The selected
span, extracted letter, failure type, and original response remain in controlled
run artifacts.

The existing weighted-mean reducer computes `genomeqa_macro`: each motif
question has weight 5 and every other question weight 1, so all 12 cells have
equal total weight. Only the macro metric is weighted; `accuracy` and
`num_solved` retain pooled accuracy and actual correct-answer counts. Existing
tagged groups emit `{subtask}/genomeqa_macro`, `{format}/genomeqa_macro`,
`{subtask}/{format}/accuracy`, and `count_{group}` metrics. The macro is declared
continuous so it does not receive a pooled-binomial uncertainty interval.

The task selects the complete population and rejects arbitrary `--limit`
subsets. Execution uses Evalchemy's existing failure and checkpoint behavior:
mostly-empty runs fail, empty responses can be retried on resume, and transport
drops can leave aggregates over fewer graded questions. Inspect expected,
generated, and graded counts before reporting a complete benchmark score.
No GenomeQA-specific runtime policy is added.

Dashboard subtask columns and public sample-access restrictions are follow-up
work in #10981. Keep GenomeQA runs out of public dashboard records until those
restrictions are in place. This adapter PR does not change Marin ingestion or
dashboard code.

The dataset and repository have no declared reuse license. Fetching them at
runtime does not settle their use terms. No upstream code, prompt, or questions
are redistributed here; tests use synthetic questions. Raw artifacts belong in
controlled storage. Do not publish question text or question text echoed by model outputs.

Issue #10981 records the completed full-benchmark Maverick validation and the
historical 100-question motif-MCQ check. Before another live evaluation, pin
the provider and effective settings and state cost and duration bounds.
