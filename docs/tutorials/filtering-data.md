# Filtering Data

Given a large pretraining dataset (e.g., FineWeb), we want to filter it down to a smaller dataset of higher-quality documents.
This guide will walk you through how to apply a custom filter to a dataset, so that you can then
train models on the filtered data.

## Prerequisites

- You must have a Ray cluster setup.
- You must have internal access to the Marin cluster.

## An Example of an Attribute Filter

Following [Dolma](https://allenai.github.io/dolma/),
Marin represents data as follows:
1. We have a set of text documents, representing the raw data.
2. We use classifiers to produce a parallel set of attributes (e.g., quality scores) for each document.
3. We then consolidate, which takes the set of documents and attributes (possibly many), a quality threshold, and produces a filtered set of documents.

See [`experiments/exp615_ensemble.py`](https://github.com/marin-community/marin/blob/main/experiments/exp615_ensemble.py) for an example filter.
This script filters FineWeb documents based on thresholding the maximum of the scores from two different fastText quality classifier models.
You can run this script using:

```bash
python experiments/exp615_ensemble.py --prefix local_store
```

## Designing Your Own Custom Filter

To understand how to design your own filter, let's take a look at what the example script is doing.
We first score the initial dataset with two previously trained fastText quality classifiers (`inference_step`), and then we define a custom attribute corresponding two the maximum of the two classifiers' scores (`ensemble_step`) using [`custom_attribute.py`](https://github.com/marin-community/marin/blob/main/marin/processing/classification/custom/custom_attribute.py).
We then consolidate the dataset (`consolidate_step`) to keep only the top 20\% of documents.

The function we use to compute the custom attribute (in this case, the maximum of the two scores) is defined in `marin.classifiers.custom.registry`. The reason we require this registry instead of directly passing functions as arguments to Executor steps is that we need Executor steps to be serializeable.
You can define your own custom attribute logic by adding a function to this registry.
