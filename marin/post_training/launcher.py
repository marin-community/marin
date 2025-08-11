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
python -m pip install -r requirements.txt
python -m pip install -U "jax[tpu]==0.4.29" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

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
<<<<<<< HEAD
        ("ray-marin-us-central1-worker-093d297d-tpu", "us-central1-a"),
=======
        ("ray-marin-us-central1-worker-4065015a-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-d79673a3-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-ca2b50da-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-f072d76d-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-1780ab4e-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-6d8fe698-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-e979dd4d-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-688be651-tpu", "us-central1-a"),
        ("ray-marin-us-east5-a-worker-71f74779-tpu", "us-east5-a"),
        ("ray-marin-us-central1-worker-877501fe-tpu", "us-central1-a"),
        ("ray-marin-us-east5-a-worker-cedab558-tpu", "us-east5-a"),
        ("ray-marin-us-central1-worker-f4b6d22b-tpu", "us-central1-a"),
        ("ray-marin-us-east5-a-worker-9a8aaa7f-tpu", "us-east5-a"),
        ("ray-marin-us-central1-worker-20da5014-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-0713ef45-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-6b36dc33-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-095ac431-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-df09da79-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-85d333be-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-a2665efb-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-af194197-tpu", "us-central1-a"),
        ("ray-marin-us-east5-a-worker-f0d9e433-tpu", "us-east5-a"),
        ("ray-marin-us-central1-worker-2be8053d-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-aea175f2-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-98689ec9-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-439d0aa0-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-6a578dc7-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-288982f3-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-e08bbbc6-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-98c7d2f8-tpu", "us-central1-a"),
        ("ray-marin-us-east5-a-worker-cf2934d6-tpu", "us-east5-a"),
        ("ray-marin-us-east5-a-worker-01f7a8ed-tpu", "us-east5-a"),
        ("ray-marin-us-central1-worker-627334fa-tpu", "us-central1-a"),
        ("ray-marin-us-central1-worker-d9ae42e3-tpu", "us-central1-a"),
>>>>>>> ffec06b9 (auto relaunch failed experiment with the lastest checkpoint)
        ("post-training-v5p-8", "us-east5-a"),
    ]

    tpu_projects = {name: create_project(name, zone) for name, zone in available_tpus}

    create_cli(
        projects=tpu_projects,
        setup=setup,
        custom_commands={"debug": debug, "check_devices": check_devices},
        launch_config_path=launch_config_path,
    )
