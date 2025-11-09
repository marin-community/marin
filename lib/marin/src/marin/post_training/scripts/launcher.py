# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from tpu_pod_launcher import TPUPodClient, TPUPodProject, create_cli

SETUP_SCRIPT = """\
cd ~/
# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
MINICONDA_INSTALLER=Miniconda3-py311_25.7.0-2-Linux-x86_64.sh
rm -f ~/$MINICONDA_INSTALLER
wget https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER -P ~/
bash ~/$MINICONDA_INSTALLER -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda create -n llama3_train python=3.11 -y
conda activate llama3_train
cd ~/llama3_train/post_training
python -m pip install -r requirements.txt
python -m pip install -U "jax[tpu]==0.4.29" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
cd ~/
rm -f ~/$MINICONDA_INSTALLER
""".strip()

CHECK_DEVICES = r"""
source ~/miniconda3/bin/activate llama3_train
python -c "import jax; print(jax.devices())"
""".strip()

USER = os.environ["USER"]


def check_devices(project: TPUPodProject, verbose: bool = False):
    project.ssh(CHECK_DEVICES, verbose=verbose)


def setup(project: TPUPodProject, verbose: bool = False):
    project.copy(verbose=verbose)
    project.ssh(SETUP_SCRIPT, verbose=verbose)
    project.ssh("mkdir ~/.config/", verbose=verbose)
    project.ssh("mkdir ~/.config/gcloud/", verbose=verbose)
    project.scp(
        f"/home/{USER}/.config/gcloud/application_default_credentials.json", "~/.config/gcloud/", verbose=verbose
    )


def debug(project: TPUPodProject, verbose: bool = False):
    import IPython

    IPython.embed()


def create_project(tpu_name: str, zone: str) -> TPUPodProject:
    return TPUPodProject(
        client=TPUPodClient(
            tpu_project="hai-gcp-models",
            tpu_zone=zone,
            user=f"{USER}",
            key_path=f"/home/{USER}/.ssh/google_compute_engine",
        ),
        tpu_name=tpu_name,
        copy_dirs=[(os.getcwd(), "~/llama3_train/")],
        working_dir="~/llama3_train/",
        copy_excludes=[".git", "__pycache__", "*.pkl", "*.json", "*.jsonl", "*.ipynb"],
        kill_commands=["sudo pkill -9 python"],
    )


if __name__ == "__main__":
    launch_config_path = os.path.join(os.path.dirname(__file__), "launch_config.json")

    available_tpus = [
        ("ray-marin-us-central1-worker-093d297d-tpu", "us-central1-a"),
        ("post-training-v5p-8", "us-east5-a"),
    ]

    tpu_projects = {name: create_project(name, zone) for name, zone in available_tpus}

    create_cli(
        projects=tpu_projects,
        setup=setup,
        custom_commands={"debug": debug, "check_devices": check_devices},
        launch_config_path=launch_config_path,
    )
