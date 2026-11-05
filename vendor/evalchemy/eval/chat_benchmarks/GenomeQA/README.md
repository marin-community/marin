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
The rendered prompt hash and source revisions are recorded in result metadata.

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

The overall `genomeqa_macro` is the equal mean of all 12 cell accuracies.
Six biological subtask scores average their BCQ and MCQ cells; format summaries
average six subtasks. Complete metrics require every expected grade. Partial
cell diagnostics retain expected, graded, missing, and correct counts. Pooled
`accuracy` is diagnostic, not the headline. Dashboard subtask views reuse the
original run without generating new answers or summing overlapping populations.

The dataset and repository have no declared reuse license. Fetching them at
runtime does not settle their use terms. No upstream code, prompt, or questions
are redistributed here; tests use synthetic questions. Raw artifacts belong in
controlled storage. Public sample browsing is disabled for GenomeQA to avoid
exposing questions or question text echoed by model outputs.

No live parity result is included. Before the 100-question Maverick comparison,
pin the OpenRouter provider and actual limits and state cost and duration bounds.
Compare correct/100 with the paper's 65% motif-MCQ result, retaining failures and
coverage; matching that number is not a release gate.
