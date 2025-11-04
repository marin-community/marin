# Direct Cache Construction

(See also [Training on Your Own Data](../Training-On-Your-Data.md) for more details on training on your own data.)

Levanter typically handles cache construction automatically, but if you have custom preprocessing logic or Ray isn't
working for you for some reason, you can directly construct a cache of preprocessed data.

You can directly construct a cache of preprocessed data without using Ray. To do so, you can use [levanter.store.SerialCacheWriter](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/store/cache.py)
to write batches directly. Here's an example:

```python
import numpy as np

from levanter.store import SerialCacheWriter

exemplar = {
    "input_ids": np.zeros((0), dtype=np.int32),
}

def process_batches():
    for i in range(0, 1000):
        yield [{"input_ids": np.array([i]) for _ in range(1000)}]

cache_dir = "gs://path/to/cache"

with SerialCacheWriter(cache_dir, exemplar) as writer:
    for batch in process_batches():
        # batch should be a list of dicts, each with keys "input_ids", "attention_mask", and "labels"
        writer.write_batch(batch)
```

In this case, `batch` should be a list of dicts, each with keys `"input_ids"`, `"attention_mask"`, and `"labels"`.
To work with `train_lm`'s `text` format, it should have an `input_ids` key that is a list of `int`s.
See the [Data Formats Reference](../reference/Data-Formats.md) for more details of other formats.

## Passthrough Tokenizers

Oftentimes, if you're using direct cache construction, you'll want to use a passthrough tokenizer. For instance,
in our music work, tokens were actually parts of a custom formatting of MIDI files and there was no actual tokenizer.

To use a cache like this, you can use the `passthrough` tokenizer:

```yaml
data:
  cache_dir: "gs://path/to/cache"
  tokenizer: "passthrough"
  vocab_size: 5567
```

The passthrough tokenizer is a special tokenizer that just passes through the input ids without any processing.
Basically, you just need to tell Levanter what the vocab size is.
