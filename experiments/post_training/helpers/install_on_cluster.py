import os

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

ACCELERATOR_TYPE = "v4-64"


@execute(TpuAcceleratorConfig(ACCELERATOR_TYPE))
@ray.remote
def main():
    os.system("pip uninstall easydel -y")
    try:
        os.system("cd ~ && git clone http://github.com/erfanzar/EasyDeL")
    except Exception:
        ...
    os.system("cd ~/EasyDeL && git pull && pip install -e .")
    os.system("pip install tensorflow tensorflow-datasets")
    os.system("pip install torch --index-url https://download.pytorch.org/whl/cpu")
    os.system("pip install jax[tpu]==0.6.0")


if __name__ == "__main__":
    main()
