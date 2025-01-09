# Overview

Our goal is to build strong base and instruction-tuned language models (like Llama 3).
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
- For model training, it uses [levanter](https://github.com/stanford-crfm/levanter),
  a Jax-based framework that's legible, scalable, and reproducible.
- For model evaluation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Where possible, we use the same data formats as [Dolma](https://github.com/allenai/dolma). Where not possible, we try to use "natural" extensions that stick to the spirit of the format.

The [quickstart eperiment](experiments/quickstart.py) provides a mini-version of
all the steps.  To run the quickstart (which should finish in less than 10
minutes, and doesn't require a GPU/TPU), run:

```bash
python experiments/quickstart.py --prefix var
```

### Snapshot tests

For HTML-to-text conversion, we have snapshot unit tests.  To add a test case,
do the following:

* Add an html file to `tests/snapshots/inputs/` that you want to test.
* Add the expected markdown output to `tests/snapshots/expected/` with the same
  name as the input file.
* Commit these files.

Pro-tip: You can copy the markdown from `process_url.py`'s output to the
expected file and edit it as needed.

If it's reasonable, try to add a unit test as well. This will help ensure that
the conversion is correct.  If you've made a change that you think is correct,
you can update the snapshots by copying `tests/snapshots/outputs/` to
`tests/snapshots/expected/`. This will overwrite the expected output with the
new output. You should review these changes before committing them.
