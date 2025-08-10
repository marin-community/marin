#!/bin/bash

# TRC recently made a change where they unceremoniously kill TPU VMs whenever they need capacity for paying customers
# Understandable, but we have to work around it.
# This script runs on a non-TPU VM (some server somewhere) and periodically relaunches the TPU VM if it's not running
# and restarts the process
# My preference would be to use pdsh for this, but we don't reliably have it on our internal cluster...

# Syntax: babysit-tpu-vm.sh <args to spin-up-vm.sh> -- command to run on the vm


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# first extract the args for just spin-up-vm.sh and pass them to the helper

CREATION_ARGS=()

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --)
      shift
      break
      ;;
    *)
      CREATION_ARGS+=("$1")
      shift
      ;;
  esac
done

source "$SCRIPT_DIR"/helpers/parse-tpu-creation-args.sh "${CREATION_ARGS[@]}"

if [ -z "$VM_NAME" ]; then
  echo "Error: VM name not set"
  exit 1
fi

if [ -z "$SSH_AUTH_SOCK" ]; then
  echo "Error: ssh-agent not running. This script needs to be run from a machine with ssh-agent running. Please run ssh-add ~/.ssh/google_compute_engine and try again"
  exit 1
fi

if [ -z "$RUN_ID" ]; then
  RUN_ID=$(bash "${SCRIPT_DIR}"/helpers/gen-id.sh)
  echo "RUN_ID not set, setting to $RUN_ID"
fi

# set the cmd args. We want to be sure everything is fully quoted when we pass it to the gcloud ssh command
# in case there are spaces in the command (or embedded quotes)
CMD_ARGS=()
for arg in "$@"; do
  # need to escape any embedded quotes using printf
  CMD_ARGS+=("$(printf '%q' "$arg")")
done

# Now turn CMD_ARGS into a single string we can pass
CMD_ARGS_STR=$(printf ' %s' "${CMD_ARGS[@]}")
CMD_ARGS_STR=${CMD_ARGS_STR:1}
CMD_ARGS_STR="RUN_ID=${RUN_ID} ${CMD_ARGS_STR}"

TRIES=0

# check if the VM is running
# if not, spin it up
# if it is, just run the command
while true; do
  # check if it's there
  gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME &> /dev/null
  if [ $? -eq 0 ]; then
    # check if it's running
    STATE=$(gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME | grep state | awk '{print $2}')
    if [ "$STATE" != "READY" ]; then
      echo "VM $VM_NAME is not in READY state, state is $STATE"
      echo "Deleting VM $VM_NAME"
      yes | gcloud compute tpus tpu-vm delete --zone $ZONE $VM_NAME
    else
      # run the command
      echo "Running command on VM $VM_NAME"
      echo "gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command='$CMD_ARGS_STR' --worker=all"
      # gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="cd ~/levanter && git checkout kaiyue/optimizer" --worker=all
      # gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="source ~/venv*/bin/activate && pip install jax[tpu]==0.4.38 jaxlib==0.4.38" --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/models/llama.py $VM_NAME:~/levanter/src/levanter/models/llama.py --worker=all
      gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/mudam.py $VM_NAME:~/levanter/src/levanter/optim/mudam.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/mudam2.py $VM_NAME:~/levanter/src/levanter/optim/mudam2.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo.py $VM_NAME:~/levanter/src/levanter/optim/shampoo.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo2.py $VM_NAME:~/levanter/src/levanter/optim/shampoo2.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo3.py $VM_NAME:~/levanter/src/levanter/optim/shampoo3.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo4.py $VM_NAME:~/levanter/src/levanter/optim/shampoo4.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo5.py $VM_NAME:~/levanter/src/levanter/optim/shampoo5.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo6.py $VM_NAME:~/levanter/src/levanter/optim/shampoo6.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo8.py $VM_NAME:~/levanter/src/levanter/optim/shampoo8.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shampoo9.py $VM_NAME:~/levanter/src/levanter/optim/shampoo9.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/shamdan.py $VM_NAME:~/levanter/src/levanter/optim/shamdan.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/mudam3.py $VM_NAME:~/levanter/src/levanter/optim/mudam3.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE src/levanter/optim/__init__.py $VM_NAME:~/levanter/src/levanter/optim/__init__.py --worker=all
      # gcloud compute tpus tpu-vm scp --zone=$ZONE config/llama2_1b_adam.yaml $VM_NAME:~/levanter/config/llama2_1b_adam.yaml --worker=all
      gcloud compute tpus tpu-vm ssh --zone=$ZONE $VM_NAME --command="$CMD_ARGS_STR" --worker=all
      EXIT_CODE=$?
      if [ $EXIT_CODE -eq 0 ]; then
        echo "Command succeeded. Exiting"
        break
      else
        echo "Command failed"
        TRIES=$((TRIES+1))
        if [ "$RETRIES" -ge 0 ]; then
          if [ $TRIES -ge "$RETRIES" ]; then
            echo "Command failed $TRIES times, exiting with $EXIT_CODE"
            break
          fi
        fi
      fi
    fi
  else
    echo "VM $VM_NAME not found, creating it"
    bash "$SCRIPT_DIR"/spin-up-vm.sh "${CREATION_ARGS[@]}"
  fi
  echo "Sleeping for 10s"
  sleep 10
done

# exit code is the exit code of the command
if [ $EXIT_CODE -eq 0 ]; then
  echo "Command succeeded"
else
  echo "Command failed too many times, ending with exit code $EXIT_CODE"
fi

# delete the VM when we're done
gcloud compute tpus tpu-vm describe --zone $ZONE $VM_NAME &> /dev/null
if [ $? -eq 0 ]; then
  echo "Deleting VM $VM_NAME"
  yes | gcloud compute tpus tpu-vm delete --zone $ZONE $VM_NAME
fi

exit $EXIT_CODE
