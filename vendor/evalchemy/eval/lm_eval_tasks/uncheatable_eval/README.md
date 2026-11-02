# UncheatableEval

This task group evaluates language-model perplexity on the July 2026
[UncheatableEval](https://huggingface.co/datasets/Jellyfish042/UncheatableEval-2026-07)
snapshot. The dataset revision is pinned in `_uncheatable_eval_template` so repeated
runs use the same documents.

Run all 15 source categories against an OpenAI-compatible completions endpoint:

```bash
eval --model local-completions --tasks uncheatable_eval \
  --model_args model=served,base_url=http://localhost:8000/v1/completions
```

Each category reports word perplexity, byte perplexity, and bits per byte (BPB) from
rolling token log-likelihoods over the dataset's `content` field. The group-level BPB
is the unweighted mean of the category scores, so large categories do not dominate the
aggregate.

The public task name `uncheatable_eval_arxiv_computer_science` selects the dataset's
`arxiv_cs` category. All other task suffixes match their dataset category names.
