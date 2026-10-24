# OpenAI MRCR

`MRCR` evaluates long-context multi-round co-reference resolution using the
corrected [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) dataset.
The dataset revision and its six parquet shards are pinned so scores do not
change when the Hub repository changes.

MRCR requires an explicit `--max_length`. It selects every example in the
published context-length bins supported by that window, interleaves the 2-, 4-,
and 8-needle cells, and routes each rendered prompt through Evalchemy's shared
endpoint context preflight before transport. The corrected dataset can have
slightly uneven cell counts after token lengths are recomputed, so selection
does not assume a fixed count per cell. `--limit` applies to the balanced order;
`--debug` selects two examples when no smaller explicit limit is provided.

```bash
uvx --from "git+https://github.com/marin-community/evalchemy[mrcr]" eval \
  --model local-chat-completions \
  --tasks MRCR \
  --max_length 131072 \
  --max_tokens 4096 \
  --model_args model=served,base_url=http://localhost:8000/v1/chat/completions
```

Results include the official nonce-gated `SequenceMatcher` score as
`mrcr_accuracy`, `prefix_hit_rate`, and a metric plus sample count for each
evaluated context-length × needle-count cell.
