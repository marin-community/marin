# Quick Start
1. Login to google cloud. It's important to set your default login to your Marin account.
```bash
gcloud auth login
gcloud config set project hai-gcp-models
gcloud auth application-default login
make dev_setup
```

2. Add your local machine's SSH key to gcloud: https://console.cloud.google.com/compute/metadata?resourceTab=sshkeys&project=hai-gcp-models&scopeTab=projectMetadata

3. Allocate an interactive node:
```bash
uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml allocate --tpu-type v5p-8
```

4. Connect to the node on IDE: Now you can connect to dev-tpu-{username} directly from VSCode/Cursor via Remote-SSH's Connect to Host feature.

# Tips
1. **Kill ghost processes:** If you encounter `RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by another process probably owned by another user`, do:
```bash
sudo rm -rf /tmp/libtpu_lockfile and sudo lsof -t /dev/vfio/* | xargs -r sudo kill -9
```

2. **Hide repeated warning messages:** If you see repeated warnings like `Could not open the log file '/tmp/tpu_logs/tpu_driver.t1v-n-796acc90-w-0.kevin.log.INFO.20250925-162309.72655': Permission denied`, you can filter them out by appending `2>&1 | grep -v "Could not"` to the end of the command. The warnings are because the ray user owns those files.
