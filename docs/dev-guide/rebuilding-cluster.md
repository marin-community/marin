# Rebuilding Ray Cluster Images

This guide walks through rebuilding the Ray cluster configuration and Docker
images. We typically attempt to rebuild images weekly to keep the base
distribution close to our source dependencies.

## Create a temporary VM

Create a VM instance for building and pushing updated cluster images:

```bash
gcloud compute instances create reboot-clusters-vm \
    --project=hai-gcp-models \
    --zone=us-south1-b \
    --machine-type=e2-standard-8 \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=748532799086-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --create-disk=auto-delete=yes,boot=yes,device-name=reboot-clusters,disk-resource-policy=projects/hai-gcp-models/regions/us-south1/resourcePolicies/default-schedule-2,image=projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2504-plucky-amd64-v20250923,mode=rw,size=100,type=pd-ssd \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
```

Configure SSH access and connect to the VM:

```bash
gcloud compute config-ssh
ssh -A reboot-clusters-vm
```

## Set up the build environment

Install required tools and clone the repository:

```bash
# Update package lists and install dependencies
sudo apt update
sudo apt install git make

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source .local/bin/env

# Clone marin repository and create a branch
git clone https://github.com/marin-community/marin
cd marin
git checkout -b $USER/update-cluster
```

## Build and deploy cluster images

Build the Docker images, tag them, and push to the registry:

```bash
# Build the cluster Docker image
make cluster_docker_build

# Tag the image with the latest version
make cluster_tag

# Push the tagged image to the container registry
make cluster_docker_push

# Update cluster configuration files with new image references
uv run ./scripts/ray/cluster.py update-configs

# Commit and push changes
git add .
git commit -m "Update cluster configuration"
git push
```

Create a pull request to update the cluster configuration in the main repository.

## Clean up

Exit the VM and delete it when finished:

```bash
exit

# Delete the temporary VM
gcloud compute instances delete reboot-clusters-vm \
    --project=hai-gcp-models \
    --zone=us-south1-b \
    --quiet
```

## Testing with a manual worker

Before committing your new branch, you can create a manual worker on an existing
cluster to test the new Docker image:

```
uv run ./scripts/ray/cluster.py --config=infra/marin-eu-west4.yaml add-worker v5litepod-4
```
