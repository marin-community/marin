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
    redis-server \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
conda create -n llama3_train python=3.10 -y
conda activate llama3_train
cd ~/llama3_train/post_training
python -m pip install uv
uv pip install -r requirements.txt
uv pip install -U "jax[tpu]==0.4.29" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
cd ~/
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
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
