# Using Dev TPUs for Testing

Using `ray_run` is great for long-running jobs, but if you are trying to debug a TPU test or memory issue, it's often faster to allocate your own TPU.
You can use the `scripts/ray/dev_tpu.py` to allocate a temporary TPU for yourself to SSH into as well as sync your local changes and automatically run commands.
You will need to setup your SSH key in `gcloud` to get started.

## Quick Start

1. Login to google cloud. It's important to set your default login to your Marin account.
```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Add your local machine's SSH key to gcloud: https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys&project=hai-gcp-models&scopeTab=projectMetadata

Your key must have a username at the end of it for this to work. It should look something like:

```
ssh-rsa ... username@MyMachine
```

This is the username you will need to use when connecting to the TPU. It's easiest to just set it to the username for your current machine: this is what `dev_tpu.py` will
use as the default user for your TPU account.

3. Allocate an interactive node:
```bash
uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml allocate --tpu-type v5p-8
```

Your TPU will boot up and synchronize your marin directory. The TPU will remain active until you close the allocate terminal (or 8 hours).

4. Connect to the node.

You can connect to the TPU in a few ways:
  - `dev_tpu.py connect` will give you an SSH terminal.
  - `dev_tpu.py execute` will sync your directory and run a remote command, for example:

```
uv run marin/scripts/ray/dev_tpu.py --tpu-name=$USER-scratch --cluster us-central1 execute -- "cd submodules/levanter && EQX_ON_ERROR=nan WANDB_MODE=offline uv run src/levanter/main/sample_lm.py --config config/sampler/sample_llama8b.yaml --n_generations 10 --n_rounds 4 --profile false"
```

  - You can also connect to dev-tpu-{username} directly from VSCode/Cursor via Remote-SSH's Connect to Host feature.

_If connecting directly, remember Dev TPUs are pre-emptible - don't forget to checkpoint your work frequently if you are making changes!_

# Tips
1. **Kill ghost processes:** If you encounter `RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by another process probably owned by another user`, do:
```bash
sudo rm -rf /tmp/libtpu_lockfile and sudo lsof -t /dev/vfio/* | xargs -r sudo kill -9
```

2. **Hide repeated warning messages:** If you see repeated warnings like `Could not open the log file '/tmp/tpu_logs/tpu_driver.t1v-n-796acc90-w-0.kevin.log.INFO.20250925-162309.72655': Permission denied`, you can filter them out by appending `2>&1 | grep -v "Could not"` to the end of the command. The warnings are because the ray user owns those files.
