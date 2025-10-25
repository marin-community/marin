# Developing Inside a GPU Docker Container

This guide assumes you've gone through the Docker setup section of the [Getting Started GPU](../Getting-Started-GPU.md) guide.
We assume you're familiar with Docker and have a basic understanding of how to use it.

## Option 1: Development Container

`.devcontainer/nvidia/devcontainer.json` in the repo can be used with [VS Code, Cursor](https://code.visualstudio.com/docs/devcontainers/containers), and [IDEA-based](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html) editors (e.g. PyCharm). [Full list of supported editors](https://containers.dev/supporting).

## Option 2: Manually Mounting Your Levanter Repo inside the Docker Container

If you are going to be adding to or extending Levanter for your own use case, these are the docker setup steps you should follow.

Clone the Levanter repository:

```bash
git clone https://github.com/stanford-crfm/levanter.git
```

Then run an interactive docker container with your levanter directory mounted as a volume. In this example, let's say your Levanter
repo is located at `/nlp/src/username/levanter`, then you would run the command below to make that directory accessible to the docker container.

```bash
sudo docker run -it --gpus=all -v /nlp/src/username/levanter:/levanter --shm-size=16g ghcr.io/nvidia/jax:levanter
```

When your container starts, the Levanter repo you cloned will be available at `/levanter`.
You should `cd` into the levanter directory and run the install command for levanter from that directory.

```bash
cd /levanter
pip install -e .
```

Now you should be able to run training jobs in this container and it will use the Levanter version you have in your mounted directory:

```bash
python src/levanter/main/train_lm.py \
    --config_path config/gpt2_nano.yaml
```

For more information on how to train models in Levanter see our [User Guide](../Getting-Started-Training.md).

### Saving Your Updated Container

After you make updates to your docker container, you may want to preserve your changes by creating a new docker image. To do so, you can detach or exit from your running container and commit these changes to a new docker images.

#### Exiting the Container
To detach from your container, but leave it running use Ctrl + P -> Ctrl + Q keys.
To stop and exit from your container, just type `exit` or use Ctrl + D.

#### Getting the Container ID
Next you need to get the ID for your container. If you run

```bash
sudo docker ps -a
```

All running and stopped containers will be listed, it should look something like this:

```
CONTAINER ID   IMAGE                         COMMAND                  CREATED             STATUS                         PORTS     NAMES
6ab837e447fd   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   26 minutes ago      Exited (1) 6 seconds ago                 gifted_ganguly
f5f5b36634f0   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   30 minutes ago      Exited (127) 26 minutes ago              great_mendel
a602487cb169   ghcr.io/nvidia/jax:levanter   "/opt/nvidia/nvidia_…"   39 minutes ago      Exited (0) 31 minutes ago                practical_lumiere
```

#### Committing the Container
You should select the `CONTAINER ID` for the container you want to preserve and run

```bash
sudo docker container commit [CONTAINER ID] [image-Name]:[image-tag]
```

If I wanted to create a new docker image called `levanter:latest` from the container `6ab837e447fd`, I would run

```bash
sudo docker container commit 6ab837e447fd levanter:latest
```

#### Using the Committed Container
Now you can start a new container using your new image and all the changes you made to the original container should still be there:

```bash
sudo docker run -it --gpus=all --shm-size=16g levanter:latest
```
