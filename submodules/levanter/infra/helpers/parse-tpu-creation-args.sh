# helper script used by spin-up-vm.sh and babysit-tpu-vm.sh


# Args we want
# VM name
# rest are flags:
# zone (default us-east1-d)
# type of box (default: v3-32
# vm image (default: tpu-vm-base)
# preemptible (default: false)
# autodelete (default: true)
# setup script (default: infra/helpers/setup-tpu-vm.sh)
# subnetwork (default: default)
# use_alpha (default: false)


# set defaults
ZONE="us-east1-d"
TYPE="v3-32"
VM_IMAGE="tpu-ubuntu2204-base"
PREEMPTIBLE=false
AUTODELETE=true
SETUP_SCRIPT="$SCRIPT_DIR/helpers/setup-tpu-vm.sh"
SUBNETWORK="default"
USE_ALPHA=false
RETRIES=-1  # how many times babysit-tpu-vm.sh should retry before giving up. -1 means infinite

if [ -z "$GIT_BRANCH" ]; then
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi

GIT_REPO=$(git config --get remote.origin.url)

# if GIT_REPO looks like an ssh url, convert it to https
if [[ "$GIT_REPO" == git@* ]]; then
  GIT_REPO=$(echo "$GIT_REPO" | sed 's/:/\//g' | sed 's/git@/https:\/\//g')
fi

# parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -z|--zone)
      ZONE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--type)
      TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--image)
      VM_IMAGE="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--preemptible)
      PREEMPTIBLE="true"
      shift # past argument
      ;;
    -a|--autodelete)
      AUTODELETE="false"
      shift # past argument
      ;;
    -s|--setup)
      SETUP_SCRIPT="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--branch)
      GIT_BRANCH="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--repo)
      GIT_REPO="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--subnetwork)
      SUBNETWORK="$2"
      shift # past argument
      shift # past value
      ;;
    --use_alpha|--use-alpha)
      USE_ALPHA="true"
      shift # past argument
      ;;
    --retries)
      RETRIES="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option, assume it's the vm name if it doesn't start with a dash
      if [[ $1 == -* ]]; then
        echo "Error: unknown option $1" >&2
        echo "Options:" >&2
        echo "  -z, --zone: zone to create the VM in (default: us-east1-d)" >&2
        echo "  -t, --type: type of VM to create (default: v3-32)" >&2
        echo "  -i, --image: VM image to use (default: tpu-vm-base)" >&2
        echo "  -p, --preemptible: use a preemptible VM (default: false)" >&2
        echo "  -a, --autodelete: delete the VM when it's done (default: true)" >&2
        echo "  -s, --setup: setup script to run on the VM (default: infra/helpers/setup-tpu-vm.sh)" >&2
        echo "  -b, --branch: git branch to use (default: current branch)" >&2
        echo "  -r, --repo: git repo to use (default: origin remote)" >&2
        echo "  -n, --subnetwork: subnetwork to use (default: default)" >&2
        echo "  --use-alpha: use gcloud alpha (default: false)" >&2
        exit 1
      fi
      # error out if we already set a name
      if [ -n "$VM_NAME" ]; then
        echo "Error: VM name already set to $VM_NAME. Got $1 as well." >&2
        exit 1
      fi
      VM_NAME="$1"
      shift # past argument
      ;;
  esac
done

# check if the branch we chose has been pushed to the remote
# if not, warn
# if it's a commit sha/short-sha (or something that looks like one), check if it's in the remote
if [[ "$GIT_BRANCH" =~ ^[0-9a-f]{7,40}$ ]]; then
  # if it's a commit, check if it's in the remote
  BRANCHES=$(git branch -r --contains "$GIT_BRANCH")
  if [ -z "$BRANCHES" ]; then
    >&2 echo "Warning: commit $GIT_BRANCH not found on remote $GIT_REPO"
  fi
else
  # get the remote branch name
  REMOTE_BRANCH=$(git ls-remote --heads origin "$GIT_BRANCH" | awk '{print $2}' | sed 's/refs\/heads\///g')
  # if it's empty, warn
  if [ -z "$REMOTE_BRANCH" ]; then
    >&2 echo "Warning: branch $GIT_BRANCH not found on remote $GIT_REPO"
  else
    # make sure it's pushed
    LOCAL_COMMIT=$(git rev-parse --short "$GIT_BRANCH")
    REMOTE_COMMIT=$(git rev-parse --short "origin/$REMOTE_BRANCH")

    if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
     >&2 echo "Warning: branch $GIT_BRANCH not pushed to remote $GIT_REPO. Local commit: $LOCAL_COMMIT, remote commit: $REMOTE_COMMIT"
    fi
  fi
fi
