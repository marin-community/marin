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
COMPOSE_FILE="${RUNTIME_DIR}/docker-compose.yml"
DOCKER_CONFIG=/etc/docker/daemon.json
HEALTH_URL="http://127.0.0.1:${LOOM_PORT}/api/health"
STARTUP_SUCCESS=/run/loom-startup-succeeded
HOME_FILES_MANIFEST=/run/loom-home-files.json
HOME_FILE_MATERIALIZER="${RUNTIME_DIR}/materialize-home-files.py"
HOME_FILE_STATE_DIR=/var/lib/loom/home-files

echo "== loom startup-script: ${LOOM_DOMAIN} =="
rm -f "$STARTUP_SUCCESS"
if [[ "$LOOM_IMAGE" != *@sha256:* ]]; then
  echo "loom startup-script: Pulumi did not supply an immutable image" >&2
  exit 1
fi

packages=()
add_apt_repository() {
  local key_url="$1"
  local key_path="$2"
  local repository="$3"
  local source_path="$4"

  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "$key_url" -o "$key_path"
  chmod a+r "$key_path"
  echo "$repository" >"$source_path"
}

if ! command -v docker >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  release="$(. /etc/os-release && echo "$VERSION_CODENAME")"
  add_apt_repository \
    https://download.docker.com/linux/debian/gpg \
    /etc/apt/keyrings/docker.asc \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian ${release} stable" \
    /etc/apt/sources.list.d/docker.list
  packages+=(docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin)
fi

if ! command -v gcloud >/dev/null 2>&1; then
  add_apt_repository \
    https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    /etc/apt/keyrings/cloud.google.asc \
    "deb [signed-by=/etc/apt/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt cloud-sdk main" \
    /etc/apt/sources.list.d/google-cloud-sdk.list
  packages+=(google-cloud-cli)
fi
if ! command -v python3 >/dev/null 2>&1; then
  packages+=(python3)
fi
if [ "${#packages[@]}" -gt 0 ]; then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

mkdir -p /etc/docker
desired_daemon_config="$(meta instance/attributes/docker-daemon-config)"
if [ ! -f "$DOCKER_CONFIG" ] || [ "$(cat "$DOCKER_CONFIG")" != "$desired_daemon_config" ]; then
  printf '%s\n' "$desired_daemon_config" >"$DOCKER_CONFIG"
  docker_config_changed=true
else
  docker_config_changed=false
fi
systemctl enable --now docker
if [ "${docker_config_changed:-false}" = true ]; then
  systemctl restart docker
fi

install -d -m 0755 "$RUNTIME_DIR"
install -d -m 0700 "$HOME_FILE_STATE_DIR"
meta instance/attributes/loom-compose >"$COMPOSE_FILE"
meta instance/attributes/loom-caddyfile >"${RUNTIME_DIR}/Caddyfile"
meta instance/attributes/loom-home-file-materializer >"$HOME_FILE_MATERIALIZER"
chmod 0700 "$HOME_FILE_MATERIALIZER"

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
docker compose -f "$COMPOSE_FILE" pull
meta instance/attributes/loom-home-files >"$HOME_FILES_MANIFEST"
docker compose -f "$COMPOSE_FILE" stop loom caddy
python3 "$HOME_FILE_MATERIALIZER" prepare \
  --manifest="$HOME_FILES_MANIFEST" \
  --image="$LOOM_IMAGE" \
  --state-dir="$HOME_FILE_STATE_DIR"
rm -f "$HOME_FILES_MANIFEST"
docker compose -f "$COMPOSE_FILE" up -d

for _ in $(seq 1 60); do
  curl -fsS "$HEALTH_URL" >/dev/null && break
  sleep 2
done
curl -fsS "$HEALTH_URL" >/dev/null

DEPLOYMENT_FILE=/run/loom-deployment.json
meta instance/attributes/loom-deployment >"$DEPLOYMENT_FILE"
docker compose -f "$COMPOSE_FILE" exec -T \
  -e "WEAVER_API=http://127.0.0.1:${LOOM_PORT}" loom \
  loom deployment apply --file - <"$DEPLOYMENT_FILE"
rm -f "$DEPLOYMENT_FILE"
touch "$STARTUP_SUCCESS"
echo "== loom startup-script done =="
