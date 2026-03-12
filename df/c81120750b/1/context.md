# Session Context

## User Prompts

### Prompt 1

Take the change set below:

```
diff --git c/experiments/pretraining_datasets/nemotron.py w/experiments/pretraining_datasets/nemotron.py
index 34099ec54..d3f4d1c27 100644
--- c/experiments/pretraining_datasets/nemotron.py
+++ w/experiments/pretraining_datasets/nemotron.py
@@ -64,7 +64,11 @@ def _get_nemotron_split_paths(split: str):
 
 
 def tokenize_nemotron(
-    *, tokenizer: str | None = None, window_size_bytes: int = 10_000_000_000
+    *,
+    tokenizer: str | None = None,
+    window_size...

### Prompt 2

is `ResourceConfig` the right place to specify `max_restarts`?

### Prompt 3

let's introduce ActorConfig

### Prompt 4

[Request interrupted by user]

### Prompt 5

continue as you were

### Prompt 6

apply request from https://github.com/marin-community/marin/pull/2960#pullrequestreview-3842965160

