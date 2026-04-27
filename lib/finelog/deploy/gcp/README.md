# finelog on Cloud Run

Cloud Run is supported for short-lived demos and single-tenant dev
deployments only. Production deployments should use GKE — see
`../k8s/`.

## Trade-off

The finelog store is single-writer: only one server may write to a given
local segment directory at a time. Cloud Run's autoscaler can recycle
instances during deploys and traffic spikes, which would corrupt the
segment cache if more than one instance ever wrote concurrently. We
clamp this with `minScale=1` / `maxScale=1`, but Cloud Run still does not
provide a true persistent local disk — the cache lives on a GCS Fuse
mount, which is slower than local SSD and has weaker consistency
semantics. GKE with a `ReadWriteOnce` PVC is the right answer for
production.

## Deploy

```bash
# 1. Build and push the image.
docker build -f lib/finelog/deploy/Dockerfile -t gcr.io/PROJECT/finelog:latest .
docker push gcr.io/PROJECT/finelog:latest

# 2. Edit cloudrun.yaml: replace PROJECT with your GCP project and
#    PROJECT-finelog* with your bucket names.

# 3. Deploy.
gcloud run services replace cloudrun.yaml --region=us-central1
```

## Network

`run.googleapis.com/ingress: internal` keeps the service off the public
internet. Callers must be on the same VPC or use a Serverless VPC
Access connector. Do not change this — the server is unauthenticated.
