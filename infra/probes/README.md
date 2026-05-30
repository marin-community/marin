# probes

Synthetic infra canary: an always-on daemon that runs three probes against Iris
and Finelog on a fixed cadence.

- `controller-ping` — `list_workers()` on the Iris controller.
- `iris-job-submit/<zone>` — submit a tiny job per zone, wait for SUCCEEDED.
- `finelog-write` — write a nonce and read it back.

Each result is logged to stdout (`probe <name>: ok|fail [<ms>ms] start=<utc>`),
written to the `infra.canary.probes` finelog namespace, and appended to a daily
JSONL that rolls up to `gs://marin-us-central1/infra/probes/dt=<date>/` at UTC
rollover.

Standalone package (own `pyproject.toml`/`uv.lock`): pulls `marin-iris`,
`marin-finelog`, `marin-rigging` from the rolling GitHub releases via
`find-links`. Bump to today's nightly with `uv lock -U` inside `infra/probes/`.

## Run

```bash
cd infra/probes
uv run python -m infra_probes --iris-endpoint http://<controller>:10000
# --zone defaults to europe-west4-b, us-west4-a
```

## Deploy

Single COS VM `infra-probes` (us-central1-b), one container, `restart=always`.

```bash
infra/probes/deploy/deploy.sh build    # build + push :sha and :latest
infra/probes/deploy/deploy.sh apply    # roll the VM to :latest
infra/probes/deploy/deploy.sh status   # VM state + recent logs
```

### One-time VM creation

```bash
PROJECT=hai-gcp-models; ZONE=us-central1-b; SA=infra-probes@${PROJECT}.iam.gserviceaccount.com

gcloud iam service-accounts create infra-probes --project=${PROJECT}

# SA needs: pull image, ship stdout to Cloud Logging, write GCS roll-ups.
gcloud artifacts repositories add-iam-policy-binding marin --project=${PROJECT} --location=us-central1 \
  --member="serviceAccount:${SA}" --role=roles/artifactregistry.reader
gcloud projects add-iam-policy-binding ${PROJECT} \
  --member="serviceAccount:${SA}" --role=roles/logging.logWriter --condition=None
gcloud storage buckets add-iam-policy-binding gs://marin-us-central1 \
  --member="serviceAccount:${SA}" --role=roles/storage.objectCreator

gcloud compute instances create-with-container infra-probes \
  --project=${PROJECT} --zone=${ZONE} --machine-type=e2-small \
  --service-account=${SA} --scopes=cloud-platform \
  --container-image=us-central1-docker.pkg.dev/${PROJECT}/marin/infra-probes:latest \
  --container-restart-policy=always \
  --container-arg="--iris-endpoint=http://iris-controller-marin.c.hai-gcp-models.internal:10000" \
  --container-mount-host-path=mount-path=/var/lib/probes,host-path=/var/lib/probes,mode=rw \
  --metadata=startup-script='#!/bin/bash
mkdir -p /var/lib/probes && chown 1000:1000 /var/lib/probes' \
  --tags=infra-probes
```

The host mount persists the JSONL across container restarts; the startup-script
makes `/var/lib/probes` writable by the uid-1000 container.
