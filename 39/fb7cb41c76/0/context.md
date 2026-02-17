# Session Context

## User Prompts

### Prompt 1

In the @lib/marin/src/marin/execution/step_runner.py let's add `.executor_info` that captures the metadata of the step. Here's an example of previous metadata:

```
{
  "name": "quickstart-tests/tokenized",
  "fn_name": "<function tokenize at 0x148c807c0>",
  "config": {
    "train_paths": [
      "/tmp/quickstart-tests/cleaned-ca7e6f"
    ],
    "validation_paths": [],
    "cache_path": "/tmp/quickstart-tests/tokenized-4ce90c",
    "tokenizer": "gpt2",
    "tags": [],
    "sample_count": null,
...

### Prompt 2

ok, let's update this, instead try to keep the same schema as existing executor_info. for config, just include the hash_attrs. for other things if we can't reproduce skip. give me updated plan.

### Prompt 3

ok

