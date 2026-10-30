# LAB-Bench public text components

Dataset and prompt attribution: Jon M. Laurent et al., FutureHouse,
[LAB-Bench](https://github.com/Future-House/LAB-Bench/tree/998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838),
[paper](https://arxiv.org/abs/2407.10362). The adapted prompt and dataset are
licensed under [CC BY-SA 4.0](LICENSE). The native `labbench` package is not installed.

`futurehouse/lab-bench` is pinned at
`5c77cec648430f30611808808861eb86f81d5eaa`. The public `train` split supplies
ProtocolQA (108), SeqQA (600), and CloningScenarios (33). The split name does
not designate these records as Marin training material. The private holdout
is excluded. ProtocolQA prepends its `protocol` field to its question.

## Evaluation protocol

Protocol `marin-labbench-no-tools-v3` evaluates generated multiple-choice answers
with reasoning allowed and no tools. It preserves the dataset's correct answer
and distractors, without adding the original harness's insufficient-information
option. Choices are shuffled with Python's `random.Random`, seeded by SHA-256
of `<run seed>:<subset>:<item id>`, independently of item order and run limits.

The prompt requests a final `Answer: X`. Scoring reuses Evalchemy's
`letter_choice` grader with HLE's extractor: it first looks for an answer-prefixed
letter, including boxed variants, then falls back to the last isolated letter.
The shared grader prefers the provider's final-answer field and removes explicit
thinking blocks before extraction. When no final field exists, normally completed
reasoning-only responses remain eligible for extraction; reasoning-only truncations
produce an empty answer and score zero. The extractor still uses the first
answer-prefixed match within the selected text.
The extracted answer is compared with the shuffled target. There is no separate
LAB-Bench parser, LLM extraction fallback, or abstention scoring. This is a
forced-choice capability evaluation; it does not reproduce the original
benchmark's uncertainty calibration or historical scores. Earlier v1 runs used
tagged answers and an added abstention choice. Version 2 scored combined reasoning
and final content, allowing an intermediate answer to override the final answer.
Version 3 corrects that shared grading behavior; retain the original scores and
identify any rescoring of saved responses separately from fresh generation.

Each component has its own Marin evaluation key and accuracy column in EvalDash.
SeqQA's 15 subtask accuracies remain diagnostic metrics within its record.
The existing reducer supplies accuracy, standard error, and sample counts.
No dedicated launcher suite or dashboard group is required.

The checked-in configurations request one response per item, temperature 0,
seed 42, and thinking enabled through the chat template. They impose no fixed
context or output cap. An explicit run/model output limit takes precedence;
otherwise Evalchemy reserves the component's measured longest prompt and shared
margins from the served context window, using the remainder for generation, as
for GPQA. The existing endpoint preflight checks requests against the actual
context window. Effective model settings and overrides are retained in run
provenance; comparisons require matching those settings. Transport failures
follow Evalchemy's existing execution-error handling.

## Running

Select the three evaluation keys with the existing launcher:

```bash
uv run python -m experiments.evaluation.cli launch \
  --model qwen3-0.6b \
  --evals lab-bench-protocolqa,lab-bench-seqqa,lab-bench-cloning \
  --limit 3 --platform gpu --accelerator H100x1 --federated_cluster cw-us-east-02a
```

The limit applies to each component; omit it to select all 741 public questions.
Capped runs are partial integration checks. These are development evaluations,
outside canonical release panels; policy adoption and component weighting are
separate decisions under [#10197](https://github.com/marin-community/marin/issues/10197).

## Deferred scope

Tools, retrieval subsets (LitQA2/SuppQA/DbQA), image subsets (FigQA/TableQA),
LABBench2 and an overall LAB-Bench score are deferred. Retrieval requires a
defined corpus/database and access policy; vision requires a multimodal route;
tool use changes the evaluated system; LABBench2 has different tasks and graders.
See [#10575](https://github.com/marin-community/marin/issues/10575) for the scope.
A full scored baseline remains pending design approval.
