#!/usr/bin/env bash
#
# GCE startup-script for the loom standalone-stack VM. Set as instance
# metadata by Pulumi; GCE runs this as root on every boot (not just the
# first), so every step below is idempotent.
#
# Reads deploy-placement metadata (repo-url, git-ref, image-mode, ar-image —
# set by Pulumi) and fetches loom's whole rendered config as ONE blob
# from Secret Manager (the LOOM_DOTENV payload is populated out of band as
# described in infra/loom/README.md), then brings up
# ../standalone/docker-compose.yml unmodified. No config field name is
# hardcoded here: this script does not parse or assemble config fields itself.
#
# Logs to the serial console and journalctl (GCE captures startup-script
# stdout/stderr under the google-startup-scripts unit) — no extra logging
# setup needed here.
#
# Bash rather than Python: this runs as the very
# first thing on a fresh VM boot, before Docker, gcloud, or uv exist — a
# Python rewrite would need to bootstrap uv first, adding fragility for no
# benefit on a boot-critical path.
set -euo pipefail

META="http://metadata.google.internal/computeMetadata/v1"
meta() { curl -sf -H "Metadata-Flavor: Google" "${META}/$1"; }

PROJECT="$(meta project/project-id)"
LOOM_DOMAIN="$(meta instance/attributes/loom-domain)"
REPO_URL="$(meta instance/attributes/repo-url)"
GIT_REF="$(meta instance/attributes/git-ref)"
IMAGE_MODE="$(meta instance/attributes/image-mode)"
AR_IMAGE="$(meta instance/attributes/ar-image 2>/dev/null || true)"
BACKUP_BUCKET="$(meta instance/attributes/backup-bucket 2>/dev/null || true)"
DOTENV_SECRET_VERSION="$(meta instance/attributes/dotenv-secret-version)"

REPO_DIR=/opt/loom
DATA_DISK_DEVICE=/dev/disk/by-id/google-loom-data
DATA_MOUNT=/mnt/loom-data

echo "== loom startup-script: domain=${LOOM_DOMAIN} image-mode=${IMAGE_MODE} =="

if [ "$IMAGE_MODE" != pull ] || [[ "$AR_IMAGE" != *@sha256:* ]]; then
  echo "loom startup-script: production requires a Pulumi-built image digest" >&2
  exit 1
fi

# Caddy requests a public certificate as soon as the stack starts. A newly
# managed Cloudflare record can lag the VM creation, so
# hold the workload until public DNS points at this VM. This guard makes a
# single Pulumi update safe even though infrastructure creation is concurrent.
EXTERNAL_IP="$(meta instance/network-interfaces/0/access-configs/0/external-ip)"
for _ in $(seq 1 80); do
  if getent ahostsv4 "$LOOM_DOMAIN" | awk '{print $1}' | grep -Fxq "$EXTERNAL_IP"; then
    echo "== ${LOOM_DOMAIN} resolves to ${EXTERNAL_IP} =="
    break
  fi
  echo "== waiting for ${LOOM_DOMAIN} to resolve to ${EXTERNAL_IP} =="
  sleep 15
done
if ! getent ahostsv4 "$LOOM_DOMAIN" | awk '{print $1}' | grep -Fxq "$EXTERNAL_IP"; then
  echo "loom startup-script: refusing to start Caddy before DNS is ready" >&2
  exit 1
fi

# ---- Docker + compose plugin ----------------------------------------------
# Docker's signed apt repo, not `curl | sh` — this is a long-lived
# internet-facing host, so root shouldn't run an unauthenticated remote
# script. Same keyring idiom as the google-cloud-cli install below.
if ! command -v docker >/dev/null 2>&1; then
  echo "== installing Docker =="
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091 # /etc/os-release is a Debian-image-provided file, not repo-tracked
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    >/etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

# ---- gcloud CLI (to read Secret Manager) -----------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
  echo "== installing google-cloud-cli =="
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    -o /etc/apt/keyrings/cloud.google.asc
  chmod a+r /etc/apt/keyrings/cloud.google.asc
  echo "deb [signed-by=/etc/apt/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" \
    >/etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update
  apt-get install -y --no-install-recommends google-cloud-cli
fi

# ---- Google Cloud Ops Agent (Prometheus receiver) --------------------------
if ! dpkg-query -W -f='${Status}' google-cloud-ops-agent 2>/dev/null | grep -q 'ok installed'; then
  echo "== installing Google Cloud Ops Agent =="
  OPS_AGENT_INSTALLER=/tmp/add-google-cloud-ops-agent-repo.sh
  curl -fsSLo "$OPS_AGENT_INSTALLER" \
    https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash "$OPS_AGENT_INSTALLER" --also-install
  rm -f "$OPS_AGENT_INSTALLER"
fi

# ---- git (to clone the repo) ----------------------------------------------
# Installed in its own command-guarded block, NOT bundled into the gcloud
# install above: on a reboot where gcloud is already present that block is
# skipped, so a git bundled there would never install and the clone below
# would fail with `git: command not found`.
if ! command -v git >/dev/null 2>&1; then
  echo "== installing git =="
  apt-get update
  apt-get install -y --no-install-recommends git
fi

# ---- optional persistent data disk for loom_home / caddy_data -------------
# Redirects Docker's entire data-root onto the separate disk, so the compose
# file's named volumes (loom_home, caddy_data, ...) land there unmodified —
# no fork of docker-compose.yml needed. See ../README.md "Durable state".
if [ -e "$DATA_DISK_DEVICE" ]; then
  if ! blkid "$DATA_DISK_DEVICE" >/dev/null 2>&1; then
    echo "== formatting data disk (first boot) =="
    mkfs.ext4 -m 0 -F "$DATA_DISK_DEVICE"
  fi
  mkdir -p "$DATA_MOUNT"
  if ! mountpoint -q "$DATA_MOUNT"; then
    mount "$DATA_DISK_DEVICE" "$DATA_MOUNT"
  fi
  # Grow the filesystem to fill the device — a no-op unless the disk was resized
  # larger (`gcloud compute disks resize`) since the last boot. This makes
  # enlarging the data disk a resize + reboot, with no manual resize2fs; ext4
  # grows online, so the mount above stays put.
  resize2fs "$DATA_DISK_DEVICE"
  if ! grep -q "^${DATA_DISK_DEVICE} " /etc/fstab; then
    echo "${DATA_DISK_DEVICE} ${DATA_MOUNT} ext4 discard,defaults,nofail 0 2" >>/etc/fstab
  fi

  mkdir -p "${DATA_MOUNT}/docker"
  if [ ! -f /etc/docker/daemon.json ] || ! grep -q "$DATA_MOUNT" /etc/docker/daemon.json 2>/dev/null; then
    echo "== pointing Docker data-root at ${DATA_MOUNT}/docker =="
    mkdir -p /etc/docker
    cat >/etc/docker/daemon.json <<EOF
{
  "data-root": "${DATA_MOUNT}/docker"
}
EOF
    systemctl restart docker
  fi
fi

# ---- get the repo, at the exact code this boot should run -----------------
# Re-run on every boot (not just clone-if-missing), so re-triggering the
# startup script (see ../README.md "Operations") actually picks up a changed
# GIT_REF/REPO_URL instead of silently keeping whatever was checked out first.
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "== initializing ${REPO_DIR} from ${REPO_URL}@${GIT_REF} =="
  mkdir -p "$REPO_DIR"
  git -C "$REPO_DIR" init
  git -C "$REPO_DIR" remote add origin "$REPO_URL"
else
  echo "== updating ${REPO_DIR} to ${REPO_URL}@${GIT_REF} =="
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
fi
# GIT_REF is normally a full release commit. Explicit fetch also makes the
# initial unmanaged metadata safe to inspect without assuming a local branch.
git -C "$REPO_DIR" fetch --depth 1 origin "$GIT_REF"
git -C "$REPO_DIR" checkout --detach --force FETCH_HEAD
git -C "$REPO_DIR" clean -fd

ENV_FILE="${REPO_DIR}/deploy/standalone/.env"

echo "== writing ${ENV_FILE} =="
umask 077
gcloud secrets versions access "$DOTENV_SECRET_VERSION" --project="$PROJECT" --secret=LOOM_DOTENV >"$ENV_FILE"
required_config=(
  LOOM_DOMAIN
  LOOM_OWNER_GITHUB
  LOOM_GITHUB_APP_ID
  LOOM_GITHUB_APP_SLUG
  LOOM_GITHUB_APP_PRIVATE_KEY
  LOOM_GITHUB_WEBHOOK_SECRET
  LOOM_GITHUB_CLIENT_ID
  LOOM_GITHUB_CLIENT_SECRET
)
for key in "${required_config[@]}"; do
  if ! grep -Eq "^${key}=.+" "$ENV_FILE"; then
    echo "loom startup-script: secret version ${DOTENV_SECRET_VERSION} is missing ${key}" >&2
    exit 1
  fi
done
# LOOM_IMAGE is deploy-placement, not a loom config field, so it isn't in the
# fetched blob. Pulumi supplies the digest produced by its host-side build.
echo "LOOM_IMAGE=${AR_IMAGE}" >>"$ENV_FILE"
# Docker-out-of-Docker: the loom container mounts this host's Docker socket (see
# ../standalone/docker-compose.yml) so sessions can `docker build`. The non-root
# app user reaches the root-owned socket by joining the host `docker` group, so
# pass its numeric gid through .env — also deploy-placement, not a config field.
# Docker is installed above, so the group exists by now.
DOCKER_GID="$(getent group docker | cut -d: -f3)"
if [ -n "$DOCKER_GID" ]; then
  echo "DOCKER_GID=${DOCKER_GID}" >>"$ENV_FILE"
fi
chmod 600 "$ENV_FILE"

# ---- nightly, online SQLite backup -----------------------------------------
# Pulumi supplies the bucket in instance metadata and grants this VM's service
# account objectCreator on it. The script uses SQLite's `.backup` API inside the
# running container, so an active WAL cannot produce a torn copy.
if [ -n "$BACKUP_BUCKET" ]; then
  install -m 0755 "${REPO_DIR}/deploy/gcp/backup-sqlite.sh" \
    /usr/local/sbin/loom-backup-sqlite
  install -m 0644 "${REPO_DIR}/deploy/gcp/loom-backup.service" \
    /etc/systemd/system/loom-backup.service
  install -m 0644 "${REPO_DIR}/deploy/gcp/loom-backup.timer" \
    /etc/systemd/system/loom-backup.timer
  systemctl daemon-reload
  systemctl enable --now loom-backup.timer
fi

# ---- bring up the stack -----------------------------------------------------
cd "${REPO_DIR}/deploy/standalone"

registry="${AR_IMAGE%%/*}"
gcloud auth configure-docker "$registry" --quiet
echo "== pulling ${AR_IMAGE} =="
docker compose pull
docker compose up -d

# ---- reconcile Pulumi-owned runtime policy ---------------------------------
# The manifest contains profile policy, Secret Manager references, and exact
# workload identities — never secret values. Apply it through Loom's REST API
# from inside the container, where the machine-local credential already exists.
DEPLOYMENT_FILE=/run/loom-deployment.json
if meta instance/attributes/loom-deployment >"$DEPLOYMENT_FILE" 2>/dev/null && [ -s "$DEPLOYMENT_FILE" ]; then
  echo "== waiting for loom before deployment reconciliation =="
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:7878/api/health >/dev/null; then
      break
    fi
    sleep 2
  done
  if ! curl -fsS http://127.0.0.1:7878/api/health >/dev/null; then
    echo "loom startup-script: loom did not become healthy for reconciliation" >&2
    exit 1
  fi
  docker compose exec -T loom loom deployment apply --file - <"$DEPLOYMENT_FILE"
  rm -f "$DEPLOYMENT_FILE"
fi

# Pulumi renders the Ops Agent receiver as instance metadata. The receiver
# scrapes Loom's loopback-only `/metrics`; its service account can only write
# telemetry and read explicitly referenced profile secrets.
OPS_AGENT_CONFIG=/etc/google-cloud-ops-agent/config.yaml
OPS_AGENT_PENDING=/run/loom-ops-agent.yaml
if meta instance/attributes/loom-ops-agent-config >"$OPS_AGENT_PENDING" 2>/dev/null && [ -s "$OPS_AGENT_PENDING" ]; then
  install -o root -g root -m 0644 "$OPS_AGENT_PENDING" "$OPS_AGENT_CONFIG"
  rm -f "$OPS_AGENT_PENDING"
  systemctl restart google-cloud-ops-agent
else
  rm -f "$OPS_AGENT_PENDING"
fi

echo "== loom startup-script done =="
