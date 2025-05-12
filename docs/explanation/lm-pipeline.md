# The Language Modeling Pipeline

Our goal is to build strong base and instruction-tuned language models.
There are many components of this pipeline, all of which are represented in Marin:


1. Curating raw sources (e.g., HTML, PDF, etc.)
2. Crawling the web for additional raw sources
3. Converting raw sources into text
4. Training quality classifers
5. Filtering the raw data with those classifiers
6. Performing deduplication to produce clean data
7. Tokenization the clean data for training
8. Training a model on the tokenized data
9. Evaluating the resulting model

Currently, we leverage the following open-source tools (thanks to the authors for making them!):

- For transforming HTML into text, we use a variety of tools including trafilatura and resiliparse.
- For data filtering, we use [fastText](https://fasttext.cc/).
- For model training, it uses [Levanter](https://github.com/stanford-crfm/levanter),
  a Jax-based framework that's legible, scalable, and reproducible.
- For model evaluation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Note that the Marin framework is agnostic to these choices and we can support other tools.

Where possible, we use the same data formats as [Dolma](https://github.com/allenai/dolma). Where not possible, we try to use "natural" extensions that stick to the spirit of the format.

The [Integration test](https://github.com/marin-community/marin/blob/main/tests/integration_test.py) provides a mini-version of
all the steps.  To run the integration test (which should finish in less than 10
minutes, and doesn't require a GPU/TPU), run:

```bash
JAX_TRACEBACK_FILTERING=off PYTHONPATH=. python tests/integration_test.py --prefix var
```
