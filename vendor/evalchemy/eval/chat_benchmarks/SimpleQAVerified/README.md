# SimpleQAVerified

`SimpleQAVerified` evaluates the 1,000 questions of Google DeepMind's
[SimpleQA Verified](https://arxiv.org/abs/2509.07968) benchmark. Its authors
filtered and relabelled OpenAI's SimpleQA to remove noisy and incorrect labels,
topical bias, and redundant questions. Candidate models receive the question
alone, without tools or retrieval context.

The MIT-licensed `simpleqa_verified.csv` is fetched from
[`google/simpleqa-verified`](https://huggingface.co/datasets/google/simpleqa-verified)
at revision `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc`. That revision includes
the publisher's 2026-03-09 correction of one question ("latitude" to
"longitude" of Lilongwe); the `codelion/SimpleQA-Verified` mirror predates it.
Each sample retains the row's `original_index` in OpenAI's SimpleQA, plus its
`topic`, `answer_type`, `multi_step`, `requires_reasoning`, and `urls` fields.

Responses are graded as correct, incorrect, or not attempted with the
SimpleQA Verified autorater prompt. The prompt is copied verbatim from version 9
of Google DeepMind's
[starter notebook](https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code),
which backs the Kaggle leaderboard; its wording matches Appendix A of the paper.
It extends the original SimpleQA prompt with numeric acceptable ranges, rules
for hedged answers, and more not-attempted examples.

Metrics and the judge-label contract match `SimpleQA`: the classification
reducer reports `accuracy`, `correct_rate`, `incorrect_rate`,
`not_attempted_rate`, `accuracy_given_attempted`, and `f1`. Deviations from the
reference notebook:

- Candidates are sampled greedily (`temperature: 0`) with a 2,048-token cap.
  The notebook uses provider defaults.
- The judge runs at `temperature: 0`. A response other than exactly `A`, `B`,
  or `C` is an unscored judge failure; the notebook searches the reply for a
  letter and defaults unparseable replies to not attempted.
- The judge model is whatever `JUDGE_MODEL` names; the leaderboard uses
  `gpt-4.1-2025-04-14`.

See [`../SimpleQA/README.md`](../SimpleQA/README.md) for judge configuration
and CLI usage.
