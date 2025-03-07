import ray
import torch_xla.distributed.xla_multiprocessing as xmp

from marin.classifiers.hf.train_classifier import HFTrainingConfig, load_dataset, train_classifier


# NOTE(chris): Important to set the PJRT_DEVICE or else sometimes it won't launch correctly because it
# does not recognize that there is a TPU device available.
@ray.remote(resources={"TPU": 8, "TPU-v6e-8-head": 1}, runtime_env={"env_vars": {"PJRT_DEVICE": "TPU"}})
def train_classifier_distributed(config: HFTrainingConfig):
    dataset = load_dataset(config.train_dataset, "train")
    dataset = dataset.train_test_split(train_size=config.train_size, seed=42)
    xmp.spawn(train_classifier, args=(config, dataset["train"], dataset["test"]))
