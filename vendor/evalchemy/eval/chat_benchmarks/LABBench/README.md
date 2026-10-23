# LAB-Bench public text components

Dataset and prompt attribution: Jon M. Laurent et al., FutureHouse,
[LAB-Bench](https://github.com/Future-House/LAB-Bench/tree/998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838),
[paper](https://arxiv.org/abs/2407.10362). The adapted prompt and dataset are
licensed under [CC BY-SA 4.0](LICENSE). This directory implements a Marin
evaluation variant; the native `labbench` package is not installed.

`futurehouse/lab-bench` is pinned at
`5c77cec648430f30611808808861eb86f81d5eaa`. The public `train` split supplies
ProtocolQA (108), SeqQA (600), and CloningScenarios (33). The split name does
not designate these records as Marin training material. The private holdout
is excluded. ProtocolQA prepends its `protocol` field to its question, matching
the native task implementation.

Protocol `marin-labbench-no-tools-v1` uses the author's zero-shot chain-of-thought
prompt and shuffled choices, including the explicit insufficient-information
option. Choice shuffling uses Python's `random.Random` seeded by SHA-256 of
`<run seed>:<subset>:<item id>`, independently of item order and run limits.
The checked-in configurations request one response per item, temperature 0,
seed 42, thinking enabled through the chat template, a 40,960-token context,
and an 8,192-token output budget. Model configuration and launch overrides
are retained in run provenance; comparisons require matching effective settings.
No model tools or LLM grader are used.

The parser accepts exactly one uppercase letter in `[ANSWER]...[/ANSWER]`,
with optional surrounding whitespace inside the tags. The letter must identify
an offered choice. Multiple tags, incomplete tags, untagged answers and invalid
letters are parse failures. For structured completions, only final content is
scored; reasoning alone cannot supply an answer. For text completions, content
before a recognized reasoning-end marker is discarded. The author's paper
describes an LLM extraction fallback; this variant deliberately omits it.

For scored model responses, let N be total responses, C correct, A explicit
abstentions, and P parse failures. Accuracy is C/N, precision is C/(N-A), and
attempted-answer coverage is (N-A)/N. Precision is 0 when N-A is zero, matching
the native evaluator's convention. Parse failures are attempted and incorrect;
they are not abstentions. The archive includes `num_total`, `num_solved`,
`num_attempted_answers`, `num_abstained`, `num_parse_failures`, `abstention_rate`
and `parse_failure_rate`. Tagged subtask metrics use the same weighted
denominators; `count_<subtask>` records each subtask's scored count.

Transport failures retain Evalchemy's infrastructure taxonomy and do not become
scored model responses. Benchmark attempted-answer coverage is distinct from
execution coverage. The source validates every selected prompt with the served
tokenizer before yielding requests: prompt + requested output + the shared
context safety margin must fit. Overflow fails preparation without dropping
questions, truncating inputs, or converting exclusions to abstentions. This
differs from the paper's treatment of context-limited Llama 3 cases. Capped
runs retain full component sizes alongside their selected/scored counts.

Each component has its own Marin evaluation key and EvalDash column. SeqQA's
15 subtask scores remain diagnostic metrics within its record. The
`lab-bench-no-tools` suite selects all three components. Selecting all three
columns in exploratory dashboard rankings gives them three entries; canonical
release-panel inclusion and weighting require a separate policy decision.

Tools, retrieval subsets (LitQA2/SuppQA/DbQA), image subsets (FigQA/TableQA),
LABBench2 and an overall LAB-Bench score are deferred. Retrieval requires a
defined corpus/database and access policy; vision requires a multimodal route;
tool use changes the evaluated system; LABBench2 has different tasks and graders.
See [#10575](https://github.com/marin-community/marin/issues/10575) for the
scope and adoption criteria. Small smoke runs establish integration evidence,
not full benchmark scores or baseline-panel readiness.

Before adoption under [#10197](https://github.com/marin-community/marin/issues/10197),
collect a full 741-question scored baseline panel with matched model settings,
inspect parse failures and abstention rates, and agree on component weighting.
That evaluation campaign is separate from this implementation's capped smoke runs.
