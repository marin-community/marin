# broadly based on https://github.com/ayaka14732/tpu-starter

# parse some arguments
# usage: ./setup-tpu-vm.sh -b|--branch <git commit or branch for levanter> -r <git repo for levanter>

if [ "$DEBUG" == "1" ]; then
  set -x
fi

REPO="https://github.com/stanford-crfm/levanter.git"
BRANCH=main

if [ "$GIT_BRANCH" != "" ]; then
  BRANCH="$GIT_BRANCH"
fi

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -b|--branch)
      BRANCH="$2"
      shift
      shift
      ;;
    -r|--repo)
      REPO="$2"
      shift
      shift
      ;;
    *)
      >&2 echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# we frequently deal with commands failing, and we like to loop until they succeed. this function does that for us
function retry {
  for i in {1..5}; do
    $@
    if [ $? -eq 0 ]; then
      break
    fi
    if [ $i -eq 5 ]; then
      >&2 echo "Error running $*, giving up"
      exit 1
    fi
    >&2 echo "Error running $*, retrying in 5 seconds"
    sleep 5
  done
}

# tcmalloc interferes with intellij remote ide
sudo patch -f -b /etc/environment << EOF
2c2
< LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
---
> #LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
EOF



# don't complain if already applied
retCode=$?
[[ $retCode -le 1 ]] || exit $retCode


# set these env variables b/c it makes tensorstore behave better
if ! grep -q TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS /etc/environment; then
  # need sudo
  echo "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60" | sudo tee -a /etc/environment > /dev/null
fi

if ! grep -q TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES /etc/environment; then
  echo "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=1024" | sudo tee -a /etc/environment > /dev/null
fi

# install python 3.11, latest git
sudo systemctl stop unattended-upgrades  # this frequently holds the apt lock
sudo systemctl disable unattended-upgrades
sudo apt remove -y unattended-upgrades
# if it's still running somehow, kill it
if [ $(ps aux | grep unattended-upgrade | wc -l) -gt 1 ]; then
  sudo kill -9 $(ps aux | grep unattended-upgrade | awk '{print $2}')
fi

# sometimes apt-get update fails, so retry a few times
retry sudo apt-get install -y software-properties-common
retry sudo add-apt-repository -y ppa:deadsnakes/ppa
retry sudo add-apt-repository -y ppa:git-core/ppa
retry sudo apt-get -qq update
retry sudo apt-get -qq install -y python3.11-full python3.11-dev git

# make absolutely sure we're installing into Python 3.11
python3.11 -m pip install --upgrade pip wheel uv || exit 1

# ensure the script-directory is on PATH
# on Ubuntu, that is usually ~/.local/bin
export PATH="$(python3.11 -m site --user-base)/bin:$PATH"

# clone levanter
if [ -d levanter ]; then
  echo "Levanter directory already exists, Assuming git repo and fetching latest changes"
  cd levanter || exit 1
  git fetch origin || exit 1
  if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
    git reset --hard "origin/$BRANCH" || exit 1
  else
    git reset --hard "$BRANCH" || exit 1
  fi
else
  git clone $REPO levanter || exit 1
  cd levanter || exit 1
  git checkout $BRANCH || exit 1
fi

# checkout the branch we want
echo "Checking out branch $BRANCH"

# install levanter
uv venv
uv sync --extra tpu
