# Overview

# TODO: redo this doc

Our goal is to build strong base and instruction-tuned language models (like Qwen 3).
The steps are roughly:

* starts with raw HTML
* processes it into text
* trains a set of quality classifers
* filters the data with those classifiers
* performs deduplication
* tokenizes the deduped data
* trains a model on the tokenized data
* evaluates the resulting model

We leverage various tools:

- For transforming HTML into text, we use a variety of tools including
  trafilatura and resiliparse.
- For data processing we use [fastText](https://fasttext.cc/).
- For model training, it uses [Levanter](https://github.com/stanford-crfm/levanter),
  a Jax-based framework that's legible, scalable, and reproducible.
- For model evaluation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Where possible, we use the same data formats as [Dolma](https://github.com/allenai/dolma). Where not possible, we try to use "natural" extensions that stick to the spirit of the format.

The [Integration test](https://github.com/stanford-crfm/marin/blob/main/tests/integration_test.py) provides a mini-version of
all the steps.  To run the integration test (which should finish in less than 10
minutes, and doesn't require a GPU/TPU), run:

```bash
JAX_TRACEBACK_FILTERING=off PYTHONPATH=. python tests/integration_test.py --prefix var
```
