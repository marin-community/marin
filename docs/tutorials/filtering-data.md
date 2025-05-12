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

For example, in [`experiments/exp615_ensemble.py`](https://github.com/marin-community/marin/blob/main/experiments/exp615_ensemble.py) we filter FineWeb documents based on ensembling the scores from two different Fasttext quality classifier models by taking their maximum.
We first score the Fineweb documents with the two Fasttext classifiers, recording each classifier's score as an attribute:
```python
for classifier_id, quality_classifier_model_path in enumerate(config.quality_classifier_model_paths):
    # run inference with each quality classifier
    inference_step = ExecutorStep(
        name=f"attributes/quality_filtering/{config.experiment_name}/{input_data_source}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=input_data_path,
            output_path=this_output_path(input_basename),
            model_name=get_model_path(quality_classifier_model_path),
            model_type="fasttext",
            attribute_name=versioned(f"{config.experiment_name}-quality_classifier-{classifier_id}"),
            runtime=RuntimeConfig(
                memory_limit_gb=12,
            ),
            task=TaskConfig(max_in_flight=500),
        ),
        pip_dependency_groups=["fasttext", "datasets", "filelock"],
    )
    inference_steps.append(inference_step)
```

We then create a new custom attribute corresponding two the maximum of the two classifiers' scores:
```python
ensemble_step = ExecutorStep(
    name=f"attributes/quality_filtering/{config.experiment_name}/{input_data_source}",
    fn=create_custom_attribute,
    config=CustomAttributeConfig(
        input_doc_path=input_data_path,
        output_attr_path=this_output_path(input_basename),
        attribute_func_name="max_quality_score",
        attribute_func_kwargs=versioned(
            {
                "score_name": "__label__hq",
                "output_attr_name": f"{config.experiment_name}-quality",
                "input_attr_names": [
                    f"{config.experiment_name}-quality_classifier-{classifier_id}"
                    for classifier_id in range(len(config.quality_classifier_model_paths))
                ],
            }
        ),
        input_attr_paths=[output_path_of(inference_step, input_basename) for inference_step in inference_steps],
    ),
)
```

Finally, we consolidate a high-quality dataset by keeping only the top 20\% of documents based on this new attribute:
```python
consolidate_step = ExecutorStep(
    name=f"documents/quality_filtering/{config.experiment_name}/{input_data_source}",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=input_data_path,
        output_path=this_output_path(input_basename),
        filters=[
            FilterConfig(
                type=versioned("classify"),
                attribute_path=output_path_of(ensemble_step, input_basename),
                name=versioned(f"{config.experiment_name}-quality"),
                label="score",
                threshold=versioned(None),
                keep_fraction=versioned(config.keep_fraction),
            ),
        ],
        ray_memory_limit_gb=12,
    ),
    pip_dependency_groups=["ddsketch"],
)
```


## Designing Your Own Custom Filter

We can create a custom attribute as an arbitrary function of existing attributes (e.g., the maximum of two existing attribute scores as in in the example above) using the `create_custom_attribute` method.
This method accepts a list of existing attributes as input along with a function used to compute the custom attribute.
The function must be defined in `marin.classifiers.custom.registry`. For example, in the example above we have
```python
@register
def max_quality_score(
    doc: list[Document],
    input_attrs: list[Attribute],
    input_attr_names: list[str],
    score_name: str,
    output_attr_name: str,
):
    """
    Take the maximum of the input attributes.
    """
    return {
        output_attr_name: {
            "score": max(
                attr["attributes"][input_attr_names[classifier_id]][score_name]
                for classifier_id, attr in enumerate(input_attrs)
            )
        }
    }
```
The reason we require this registry instead of directly passing functions as arguments to Executor steps is that we need Executor steps to be serializeable.
You can define your own custom attribute logic by adding a function to this registry using the provided decorator.

## Next Steps

After filtering your dataset, the next step is to train a model on the filtered data. See our guide on [how to train a language model](./train-an-lm.md).
