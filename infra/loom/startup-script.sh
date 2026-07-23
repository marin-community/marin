#!/usr/bin/env bash
# Reconcile the small host-side shell around the Pulumi-built Loom image.
set -euo pipefail

META=http://metadata.google.internal/computeMetadata/v1
meta() { curl -fsS -H "Metadata-Flavor: Google" "${META}/$1"; }

PROJECT="$(meta project/project-id)"
LOOM_DOMAIN="$(meta instance/attributes/loom-domain)"
LOOM_IMAGE="$(meta instance/attributes/loom-image)"
DOTENV_SECRET_VERSION="$(meta instance/attributes/dotenv-secret-version)"
DOTENV_SECRET_ID="$(meta instance/attributes/dotenv-secret-id)"
LOOM_PORT="$(meta instance/attributes/loom-port)"
RUNTIME_DIR=/opt/loom
DATA_DISK_DEVICE="/dev/disk/by-id/google-$(meta instance/attributes/data-disk-device)"
DATA_MOUNT=/mnt/loom-data

echo "== loom startup-script: ${LOOM_DOMAIN} =="
if [[ "$LOOM_IMAGE" != *@sha256:* ]]; then
  echo "loom startup-script: Pulumi did not supply an immutable image" >&2
  exit 1
fi

packages=()
if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091
  release="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian ${release} stable" \
    >/etc/apt/sources.list.d/docker.list
  packages+=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)
fi

if ! command -v gcloud >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg -o /etc/apt/keyrings/cloud.google.asc
  chmod a+r /etc/apt/keyrings/cloud.google.asc
  echo "deb [signed-by=/etc/apt/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" \
    >/etc/apt/sources.list.d/google-cloud-sdk.list
  packages+=(google-cloud-cli)
fi
if [ "${#packages[@]}" -gt 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

if [ ! -e "$DATA_DISK_DEVICE" ]; then
  echo "loom startup-script: durable data disk is not attached" >&2
  exit 1
fi
if filesystem_type="$(blkid -p -s TYPE -o value "$DATA_DISK_DEVICE")"; then
  if [ "$filesystem_type" != ext4 ]; then
    echo "loom startup-script: durable data disk uses unexpected filesystem ${filesystem_type}" >&2
    exit 1
  fi
else
  blkid_status=$?
  if [ "$blkid_status" -ne 2 ]; then
    echo "loom startup-script: could not inspect durable data disk (blkid ${blkid_status})" >&2
    exit 1
  fi
  mkfs.ext4 -m 0 "$DATA_DISK_DEVICE"
fi
mkdir -p "$DATA_MOUNT"
if mountpoint -q "$DATA_MOUNT"; then
  data_disk_mounted=false
else
  mount "$DATA_DISK_DEVICE" "$DATA_MOUNT"
  data_disk_mounted=true
fi
resize2fs "$DATA_DISK_DEVICE"
grep -q "^${DATA_DISK_DEVICE} " /etc/fstab || \
  echo "${DATA_DISK_DEVICE} ${DATA_MOUNT} ext4 discard,defaults,nofail 0 2" >>/etc/fstab
mkdir -p "${DATA_MOUNT}/docker" /etc/docker
desired_daemon_config="$(printf '{\n  "data-root": "%s/docker"\n}\n' "$DATA_MOUNT")"
if [ ! -f /etc/docker/daemon.json ] || [ "$(cat /etc/docker/daemon.json)" != "$desired_daemon_config" ]; then
  printf '%s\n' "$desired_daemon_config" >/etc/docker/daemon.json
  docker_config_changed=true
else
  docker_config_changed=false
fi
systemctl enable --now docker
if [ "${docker_config_changed:-false}" = true ] || [ "$data_disk_mounted" = true ]; then
  systemctl restart docker
fi

install -d -m 0755 "$RUNTIME_DIR"
meta instance/attributes/loom-compose >"${RUNTIME_DIR}/docker-compose.yml"
meta instance/attributes/loom-caddyfile >"${RUNTIME_DIR}/Caddyfile"

ENV_FILE="${RUNTIME_DIR}/.env"
umask 077
gcloud secrets versions access "$DOTENV_SECRET_VERSION" \
  --project="$PROJECT" --secret="$DOTENV_SECRET_ID" >"$ENV_FILE"
required_config=(
  LOOM_OWNER_GITHUB
  LOOM_GITHUB_APP_ID
  LOOM_GITHUB_APP_SLUG
  LOOM_GITHUB_APP_PRIVATE_KEY
  LOOM_GITHUB_WEBHOOK_SECRET
  LOOM_GITHUB_CLIENT_ID
  LOOM_GITHUB_CLIENT_SECRET
)
for key in "${required_config[@]}"; do
  grep -Eq "^${key}=.+" "$ENV_FILE" || {
    echo "loom startup-script: secret version ${DOTENV_SECRET_VERSION} is missing ${key}" >&2
    exit 1
  }
done
sed -i -E '/^(LOOM_DOMAIN|LOOM_IMAGE|LOOM_PORT|DOCKER_GID)=/d' "$ENV_FILE"
{
  printf '\n'
  printf 'LOOM_IMAGE=%s\nLOOM_PORT=%s\nDOCKER_GID=%s\n' \
    "$LOOM_IMAGE" "$LOOM_PORT" "$(getent group docker | cut -d: -f3)"
  printf 'LOOM_DOMAIN=%s\n' "$LOOM_DOMAIN"
} >>"$ENV_FILE"
chmod 0600 "$ENV_FILE"

registry="${LOOM_IMAGE%%/*}"
gcloud auth configure-docker "$registry" --quiet
docker compose -f "${RUNTIME_DIR}/docker-compose.yml" pull
docker compose -f "${RUNTIME_DIR}/docker-compose.yml" up -d

for _ in $(seq 1 60); do
  curl -fsS "http://127.0.0.1:${LOOM_PORT}/api/health" >/dev/null && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${LOOM_PORT}/api/health" >/dev/null

DEPLOYMENT_FILE=/run/loom-deployment.json
meta instance/attributes/loom-deployment >"$DEPLOYMENT_FILE"
docker compose -f "${RUNTIME_DIR}/docker-compose.yml" exec -T loom \
  loom deployment apply --file - <"$DEPLOYMENT_FILE"
rm -f "$DEPLOYMENT_FILE"
echo "== loom startup-script done =="
