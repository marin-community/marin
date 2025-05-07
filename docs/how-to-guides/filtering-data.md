# Filtering Data

This guide will walk you through how to apply a custom filter to a dataset, so that you can then
train models on the filtered data.

## An Example of an Attribute Filter

Marin organizes data into collections of documents and attributes. One example of an attribute is a quality classifier score for a particular classifier. We can filter collections of document based on various (combinations of) attributes in order to obtain higher-quality documents.

See ['experiments/filtering.py'] for an example filter. This script filters Fineweb documents based on thresholding the maximum of the scores from two different Fasttext quality classifier models. You can run this script using:

```bash
python  experiments/exp615.py
```

## Designing Your Own Custom Filter

To understand how to design your own filter, let's take a look at what the example script is doing.
We first score the initial dataset with two previously trained Fasttext quality classifiers (`inference_step`), and then we define a custom attribute corresponding two the maximum of the two classifiers' scores (`ensemble_step`) using ['custom_attribute.py'](https://github.com/stanford-crfm/marin/blob/main/marin/processing/classification/custom/custom_attribute.py).
We then consolidate the dataset (`consolidate_step`) to keep only the top 20\% of documents.

The function we use to compute the custom attribute (in this case, the maximum of the two scores) is defined in `marin.classifiers.custom.registry`. The reason we require this registry instead of directly passing functions as arguments to Executor steps is that we need Executor steps to be serializeable.
You can define your own custom attribute logic by adding a function to this registry.
