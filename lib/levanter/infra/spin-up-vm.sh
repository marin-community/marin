#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. "${SCRIPT_DIR}"/helpers/parse-tpu-creation-args.sh "$@"

# error out if we didn't set a name
if [ -z "$VM_NAME" ]; then
  echo "Error: VM name not set"
  exit 1
fi

# Determine the correct gcloud command based on USE_ALPHA flag
GCLOUD_CMD="gcloud compute"
if [ "$USE_ALPHA" = true ]; then
  GCLOUD_CMD="gcloud alpha compute"
fi

# first delete if we're supposed to
if [ "$AUTODELETE" = "true" ]; then
  # check if it's there
  $GCLOUD_CMD tpus tpu-vm describe --zone $ZONE $VM_NAME &> /dev/null
  if [ $? -eq 0 ]; then
    echo "Deleting existing VM $VM_NAME"
    $GCLOUD_CMD tpus tpu-vm delete --zone $ZONE $VM_NAME
  fi
fi

# if ssh-agent isn't running, complain
if [ -z "$SSH_AUTH_SOCK" ]; then
  echo "Error: ssh-agent not running"
  exit 1
fi

# create the vm
# spin loop until we get a good error code
echo "Creating VM $VM_NAME"
# create the command. note that --preemptible doesn't accept a value, so just append it if we want it
CMD="$GCLOUD_CMD tpus tpu-vm create $VM_NAME \
  --zone=$ZONE \
  --accelerator-type=$TYPE \
  --version=$VM_IMAGE"

# if network isn't 'default', set it
if [ "$NETWORK" != "default" ]; then
  CMD="$CMD --network=$NETWORK"
fi

if [ "$PREEMPTIBLE" = true ]; then
  CMD="$CMD --preemptible"
fi
echo "Running command: $CMD"

while ! $CMD; do
  echo "Error creating VM, retrying in 5 seconds"
  sleep 5
done

echo "Giving the VM a few seconds to start up"
sleep 15

echo "Adding ssh keys just in case..."
echo ssh-add ~/.ssh/google_compute_engine
ssh-add ~/.ssh/google_compute_engine

# upload the setup script
SETUP_SCRIPT_NAME=$(basename $SETUP_SCRIPT)
# note that gcloud scp doesn't always work... so we do it a few times to just be sure
for i in {1..5}; do
  echo "Uploading $SETUP_SCRIPT to VM $VM_NAME"
  $GCLOUD_CMD tpus tpu-vm scp --zone=$ZONE $SETUP_SCRIPT $VM_NAME:~/ --worker=all
  # check to see if the file exists on all nodes
  if $GCLOUD_CMD tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="ls ~/$SETUP_SCRIPT_NAME" --worker=all; then
    break
  fi
  if [ 5 -eq $i ]; then
    echo "Error uploading ${SETUP_SCRIPT_NAME}, giving up. Note that the machine is still (probably) running"
    exit 1
  fi
  echo "Error uploading ${SETUP_SCRIPT_NAME}, retrying in 5 seconds"
  sleep 5
done

# run the setup script
for i in {1..5}; do
  $GCLOUD_CMD tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="bash ~/$SETUP_SCRIPT_NAME --branch ${GIT_BRANCH} --repo ${GIT_REPO} > setup.out" --worker=all
  if [ $? -eq 0 ]; then
    break
  fi
  if [ 5 -eq $i ]; then
    echo "Error running ${SETUP_SCRIPT_NAME}, giving up. Note that the machine is still (probably) running"
    exit 1
  fi
  echo "Error running ${SETUP_SCRIPT_NAME}, retrying in 5 seconds"
  sleep 5
done

# print out the IP addresses
echo "VM $VM_NAME IP addresses:"
$GCLOUD_CMD tpus tpu-vm describe --zone $ZONE $VM_NAME | awk '/externalIp: (.*)/ {print $2}'
