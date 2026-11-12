# SimpleQA

`SimpleQA` evaluates all 4,326 short, fact-seeking questions from OpenAI's
[SimpleQA benchmark](https://openai.com/index/introducing-simpleqa/). Candidate
models receive the original question without retrieval context. Responses are
graded as correct, incorrect, or not attempted with the canonical SimpleQA
classifier prompt and Evalchemy's normalized `JUDGE_MODEL`, `JUDGE_BASE_URL`,
and `JUDGE_API_KEY` configuration.

The canonical OpenAI `simple_qa_test_set.csv` is fetched from the
[basicv8vc/SimpleQA mirror](https://huggingface.co/datasets/basicv8vc/SimpleQA)
at revision `e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94` through Evalchemy's
dataset cache. This mirror is byte-identical to the previously bundled CSV.
Its SHA-256 digest is
`feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`.
The benchmark and classifier are adapted from OpenAI's MIT-licensed
[`simple-evals`](https://github.com/openai/simple-evals) reference and Harbor's
[SimpleQA adapter](https://github.com/harbor-framework/harbor/tree/main/adapters/simpleqa).

`SimpleQAMini` is a separate Evalchemy task containing 500 questions. It uses
the reference evaluator's deterministic subset rule:
`random.Random(0).sample(full_dataset, 500)`. The source row index is retained
in each sample record so the subset remains auditable.

Run either task with the standard Evalchemy CLI:

```bash
export JUDGE_API_KEY=...
eval --model local-chat-completions --tasks SimpleQA,SimpleQAMini \
  --model_args model=served,base_url=http://localhost:8000/v1/chat/completions
```
