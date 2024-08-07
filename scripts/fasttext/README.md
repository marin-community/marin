# fastText training

## Create the dataset
Use `create_dataset.py` to create multi-label fastText formatted sharded dataset. Associate labels to documents using
a config file.

```bash
python create_dataset.py --config_path {PATH_TO_CONFIG_FILE}
```
e.g., including
```yaml
...
data_cfgs:
  - path: gs://{BUCKET}
    dataset: reddit/v0
    experiment: experiment-0
    labels:
      - great
...
```
in the config file will associate the label `great` with all documents in `gs://{BUCKET}/documents/experiment-0/reddit/v0`.
Eventually, we might want to automatically determine labels as a function of some existing attributes. 
See `attribute_to_dataset.py` for a minimal example of what this might look like.

## Train the model
Use `train_model.py` to train a fastText model on a multi-label dataset (e.g., the output of `create_dataset.py`).

```bash
python train_model.py --config_path {PATH_TO_CONFIG_FILE}
```
e.g., including
```yaml
...
data_cfgs:
  - path: gs://{BUCKET}
    experiment: experiment-1
...
```
in the config file will concatenate all shards in `gs://{BUCKET}/classifiers/experiment-1/data` into training and validation
datasets, train a fastText model, and save the model to `gs://{BUCKET}/classifiers/experiment-1/`.