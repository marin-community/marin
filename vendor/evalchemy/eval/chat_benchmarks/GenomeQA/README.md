# GenomeQA

This adapter evaluates the original 5,200 questions from
[GenomeQA](https://arxiv.org/abs/2604.05774). Marin issue
[#10981](https://github.com/marin-community/marin/issues/10981) records the
protocol choices and alternatives. Registry availability does not admit the
benchmark into Marin's official evaluation policy.

`GenomeQA` loads the pinned HF `binary` and `mcq` files. `GenomeQAMotifMCQ`
uses the same implementation for all 100 motif MCQs, the agreed first-pass
comparison. Neither task runs tools, retrieves context, shuffles options, or
optimizes prompts. Question identities include format, subtask, and original ID.

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

The grader selects final content, strips supported thinking markers, and uses
the last `### Answer` section. Without that heading it requires an explicit
`Answer: X`. Conflicting or invalid letters fail. Empty answers, explicit provider
refusals, and output-limit truncations receive zero; transport failures remain
ungraded. The selected span, extracted letter, failure type, and original response
remain in controlled run artifacts.

The existing weighted-mean reducer computes `genomeqa_macro`: each motif
question has weight 5 and every other question weight 1, so all 12 cells have
equal total weight. Only the macro metric is weighted; `accuracy` and
`num_solved` retain pooled accuracy and actual correct-answer counts. Existing
tagged groups emit `{subtask}/genomeqa_macro`, `{format}/genomeqa_macro`,
`{subtask}/{format}/accuracy`, and `count_{group}` metrics. The macro is declared
continuous so it does not receive a pooled-binomial uncertainty interval.

Both tasks require their complete population and reject arbitrary `--limit`
subsets. Completed empty responses score zero and remain checkpointed on resume;
transport failures remain retryable. Any missing grade suppresses
aggregate publication; the shared runtime retains individual samples and failure
counts for inspection and resume. The small runtime policy opt-ins preserve the
default behavior of other benchmarks.

Dashboard subtask columns and public sample-access restrictions are follow-up
work in #10981. Keep GenomeQA runs out of public dashboard records until those
restrictions are in place. This adapter PR does not change Marin ingestion or
dashboard code.

The dataset and repository have no declared reuse license. Fetching them at
runtime does not settle their use terms. No upstream code, prompt, or questions
are redistributed here; tests use synthetic questions. Raw artifacts belong in
controlled storage. Do not publish question text or question text echoed by model outputs.

No live parity result is included. Before the 100-question Maverick comparison,
pin the OpenRouter provider and actual limits and state cost and duration bounds.
Compare correct/100 with the paper's 65% motif-MCQ result, retaining failures and
coverage; matching that number is not a release gate.
