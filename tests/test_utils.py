import ray
from marin.utils import remove_tpu_lockfile_on_exit


def setup_module(module):
    ray.init("local", num_cpus=8, ignore_reinit_error=True)


def teardown_module(module):
    ray.shutdown()

def test_remove_tpu_lockfile_on_exit_works_with_ray_remote():
    @ray.remote
    @remove_tpu_lockfile_on_exit
    def test_fn():
        return 1

    assert ray.get(test_fn.remote()) == 1


