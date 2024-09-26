import logging

import ray

from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig

from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

@ray.remote(num_cpus=0.1, runtime_env={"pip": ["levanter @ git+https://github.com/stanford-crfm/levanter.git@tweaks",
                                               "jax[tpu]==0.4.30",
                                               "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20240617+default-py3-none-any.whl"]})
@remove_tpu_lockfile_on_exit
def run_levanter_train_lm(config: TrainLmConfig, tpu_type: str):
    suppress_ray_config(config)
    from levanter.infra.ray_tpu import run_on_pod_resumable

    @ray.remote
    def run_on_pod_resumable_fn():
        import jax
        print(jax.devices("tpu"))
        # train_lm.main(config)

    run_on_pod_resumable(run_on_pod_resumable_fn, tpu_type)


def suppress_ray_config(config):
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        # TODO: hacky mutation, but there are no lenses in python i think
        config.trainer.ray.auto_start_cluster = False
        config.trainer.ray.start_workers = False
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        config.trainer.ray.start_workers = False



if __name__ == "__main__":
    ray.init()
    default_config = TrainLmConfig()
    default_config.data.cache_dir = "/tmp/levanter_cache"
    default_config.data.id = "dlwh/wikitext_103_detokenized"
    default_config.trainer.tracker = ()
    default_config.trainer.require_accelerator = True
    from levanter.models import gpt2
    default_config.model = gpt2.Gpt2Config(
        num_heads=4,
        hidden_dim=128
    )
    default_config.trainer.train_batch_size = 8
    ray.get(run_levanter_train_lm.remote(default_config, "v4-16"))
    ray.shutdown()