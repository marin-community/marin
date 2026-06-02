#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Install + configure Kueue on a CoreWeave CKS cluster for Iris gang admission.
#
# What it does (idempotent; helm upgrade --install):
#   1. Adds the CoreWeave helm repo and installs the `cks-kueue` chart into the
#      kueue-system namespace.
#   2. Creates the CoreWeave Topology CRs (infiniband + multinode-nvlink-ib) so
#      topology-aware scheduling can honor the podset-topology annotations Iris
#      stamps (backend.coreweave.cloud/leafgroup, ds.coreweave.com/nvlink.domain).
#   3. Enables the plain-Pod integration (NOT on by default in the chart) so
#      Kueue gang-admits Iris's scheduling-gated pod groups, SCOPED to an
#      explicit namespace allowlist — the CKS cluster is shared, so the pod
#      webhook must not touch other tenants' namespaces.
#   4. (optional, --with-queues) Creates the cluster-scoped, admin-owned
#      ResourceFlavor + ClusterQueue (the quota). The namespaced LocalQueue is
#      NOT created here: Iris reconciles its own LocalQueue at controller start
#      (K8sControllerManager.ensure_kueue_queues), binding it to this
#      ClusterQueue via kubernetes_provider.kueue.cluster_queue.
#
# SAFE BY DEFAULT: prints the helm/kubectl plan and a server-side dry-run, then
# stops. Pass --apply to mutate the cluster. This touches a SHARED CoreWeave
# cluster — review the dry-run and the namespace allowlist before applying.
#
# Requires: helm >= 3.8, kubectl. Point at the cluster with --kubeconfig and/or
# --context (or the usual KUBECONFIG env var).
#
# Why this exists / what the CoreWeave docs leave out:
#   https://docs.coreweave.com/products/cks/clusters/coreweave-charts/kueue
#   documents the repo, the install, and `topologies:` but NOT how to enable the
#   plain-Pod integration that Iris's direct provider relies on. That is the
#   `integrations.frameworks: ["pod"]` + scoped `podOptions.namespaceSelector`
#   block this script injects via the wrapped upstream kueue subchart's
#   managerConfig.

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults / config
# --------------------------------------------------------------------------
REPO_NAME="coreweave"
REPO_URL="https://charts.core-services.ingress.coreweave.com"
CHART="${REPO_NAME}/cks-kueue"
CHART_VERSION=""            # empty => latest; pin with --chart-version X.Y.Z
RELEASE="kueue"
OPERATOR_NS="kueue-system"

KUBECONFIG_ARG=""
CONTEXT_ARG=""
APPLY=0
WITH_QUEUES=0

# Namespaces the Kueue pod webhook should manage (gang-admit pods in). Defaults
# to a single Iris namespace; override/extend with repeated --namespace flags.
TENANT_NAMESPACES=()

# ClusterQueue nominal quota used only with --with-queues. Sized for the caller
# to edit; these are NOT auto-derived from NodePool capacity.
CQ_CPU="384"               # e.g. 3 x 128-core H100 nodes
CQ_MEMORY="1536Gi"
CQ_GPU="24"                # e.g. 3 x 8 H100
CLUSTER_QUEUE="iris-cq"    # ClusterQueue name Iris's LocalQueue binds to

usage() {
  cat <<'EOF'
Usage: install_kueue_coreweave.sh [options]

  --kubeconfig PATH     kubeconfig to use (else $KUBECONFIG / ~/.kube/config)
  --context NAME        kube context to target
  --namespace NS        Iris namespace the pod webhook manages (repeatable;
                        default: iris). Pod gang admission is scoped to these.
  --chart-version VER   pin the cks-kueue chart version (default: latest)
  --release NAME        helm release name (default: kueue)
  --with-queues         also create the cluster-scoped ResourceFlavor + ClusterQueue
  --cluster-queue NAME  ClusterQueue name for --with-queues (default: iris-cq)
  --cq-cpu N            ClusterQueue cpu nominal quota   (default: 384)
  --cq-memory SIZE      ClusterQueue memory nominal quota (default: 1536Gi)
  --cq-gpu N            ClusterQueue nvidia.com/gpu quota (default: 24)
  --apply               actually mutate the cluster (default: dry-run only)
  -h, --help            show this help

Examples:
  # Review the plan against a cluster (no changes):
  install_kueue_coreweave.sh --kubeconfig ~/.kube/coreweave-iris --namespace iris-ci

  # Apply, scoping pod admission to two Iris namespaces, and create queues:
  install_kueue_coreweave.sh --kubeconfig ~/.kube/coreweave-iris \
      --namespace iris-ci --namespace iris-prod --with-queues --apply
EOF
}

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kubeconfig)     KUBECONFIG_ARG="$2"; shift 2 ;;
    --context)        CONTEXT_ARG="$2"; shift 2 ;;
    --namespace)      TENANT_NAMESPACES+=("$2"); shift 2 ;;
    --chart-version)  CHART_VERSION="$2"; shift 2 ;;
    --release)        RELEASE="$2"; shift 2 ;;
    --with-queues)    WITH_QUEUES=1; shift ;;
    --cluster-queue)  CLUSTER_QUEUE="$2"; shift 2 ;;
    --cq-cpu)         CQ_CPU="$2"; shift 2 ;;
    --cq-memory)      CQ_MEMORY="$2"; shift 2 ;;
    --cq-gpu)         CQ_GPU="$2"; shift 2 ;;
    --apply)          APPLY=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ${#TENANT_NAMESPACES[@]} -eq 0 ]]; then
  TENANT_NAMESPACES=("iris")
fi

# Shared flags threaded into every helm/kubectl invocation.
KUBE_FLAGS=()
[[ -n "$KUBECONFIG_ARG" ]] && KUBE_FLAGS+=(--kubeconfig "$KUBECONFIG_ARG")
[[ -n "$CONTEXT_ARG" ]] && KUBE_FLAGS+=(--kube-context "$CONTEXT_ARG")
# kubectl spells the context flag differently than helm.
KCTL_FLAGS=()
[[ -n "$KUBECONFIG_ARG" ]] && KCTL_FLAGS+=(--kubeconfig "$KUBECONFIG_ARG")
[[ -n "$CONTEXT_ARG" ]] && KCTL_FLAGS+=(--context "$CONTEXT_ARG")

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }
require helm
require kubectl

# --------------------------------------------------------------------------
# Render the namespaceSelector that scopes pod gang admission to Iris namespaces.
# matchExpressions In <list> => the webhook only gates pods in those namespaces;
# every other tenant on the shared cluster is untouched.
# --------------------------------------------------------------------------
ns_selector_yaml() {
  local indent="$1"
  printf '%smatchExpressions:\n' "$indent"
  printf '%s  - key: kubernetes.io/metadata.name\n' "$indent"
  printf '%s    operator: In\n' "$indent"
  printf '%s    values:\n' "$indent"
  local ns
  for ns in "${TENANT_NAMESPACES[@]}"; do
    printf '%s      - %s\n' "$indent" "$ns"
  done
}

# --------------------------------------------------------------------------
# Build the helm values file.
#
# - topologies: the two CoreWeave-documented Topology CRs. Iris's preferred
#   "pool" topology rides on backend.coreweave.cloud/leafgroup and the required
#   "nvlink"/"tpu-name" topology on ds.coreweave.com/nvlink.domain — both are
#   levels here, so TAS can satisfy the annotations.
# - kueue.managerConfig.controllerManagerConfigYaml: the upstream kueue
#   subchart's Configuration. We enable the "pod" framework (gang admission for
#   plain pods) alongside "batch/job" and scope it to the Iris namespaces.
#   manageJobsWithoutQueueName stays false so only queue-labeled (Iris) pods are
#   gated. The TopologyAwareScheduling feature gate is left to the chart default
#   (already --feature-gates=TopologyAwareScheduling=true).
# --------------------------------------------------------------------------
VALUES_FILE="$(mktemp -t kueue-values.XXXXXX.yaml)"
trap 'rm -f "$VALUES_FILE"' EXIT

{
  cat <<'EOF'
topologies:
  - name: infiniband
    levels:
      - backend.coreweave.cloud/fabric
      - backend.coreweave.cloud/superpod
      - backend.coreweave.cloud/leafgroup
      - kubernetes.io/hostname
  - name: multinode-nvlink-ib
    levels:
      - backend.coreweave.cloud/fabric
      - backend.coreweave.cloud/superpod
      - backend.coreweave.cloud/leafgroup
      - ds.coreweave.com/nvlink.domain
      - kubernetes.io/hostname

kueue:
  enableKueueViz: false
  # NB: the chart already enables --feature-gates=TopologyAwareScheduling=true by
  # default (its controllerManager.featureGates is a list, not the map you might
  # expect), so we do NOT override it — doing so would clobber that default.
  managerConfig:
    controllerManagerConfigYaml: |-
      apiVersion: config.kueue.x-k8s.io/v1beta1
      kind: Configuration
      health:
        healthProbeBindAddress: :8081
      metrics:
        bindAddress: :8080
      webhook:
        port: 9443
      manageJobsWithoutQueueName: false
      internalCertManagement:
        enable: true
        webhookServiceName: kueue-webhook-service
        webhookSecretName: kueue-webhook-server-cert
      integrations:
        frameworks:
          - "batch/job"
          - "pod"
        podOptions:
          namespaceSelector:
EOF
  ns_selector_yaml "            "
} > "$VALUES_FILE"

log "Rendered helm values -> $VALUES_FILE"
sed 's/^/    /' "$VALUES_FILE"
log "Pod gang admission scoped to namespaces: ${TENANT_NAMESPACES[*]}"

# --------------------------------------------------------------------------
# Cluster-scoped, admin-owned quota objects (defined before use so the dry-run
# branch can print them too). The namespaced LocalQueue is intentionally NOT
# here — Iris reconciles that at controller start.
# --------------------------------------------------------------------------
render_queues() {
  cat <<EOF
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ResourceFlavor
metadata:
  name: cw-ib
spec:
  # Tie the flavor to the IB Topology so podset-topology annotations resolve.
  topologyName: infiniband
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: ${CLUSTER_QUEUE}
spec:
  namespaceSelector: {}
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: cw-ib
          resources:
            - name: cpu
              nominalQuota: "${CQ_CPU}"
            - name: memory
              nominalQuota: "${CQ_MEMORY}"
            - name: nvidia.com/gpu
              nominalQuota: "${CQ_GPU}"
EOF
}

# --------------------------------------------------------------------------
# helm repo + install/upgrade
# --------------------------------------------------------------------------
log "Adding/updating helm repo $REPO_NAME ($REPO_URL)"
helm repo add "$REPO_NAME" "$REPO_URL" >/dev/null
helm repo update "$REPO_NAME" >/dev/null

HELM_ARGS=(upgrade --install "$RELEASE" "$CHART"
  --namespace "$OPERATOR_NS" --create-namespace
  --values "$VALUES_FILE" "${KUBE_FLAGS[@]}")
[[ -n "$CHART_VERSION" ]] && HELM_ARGS+=(--version "$CHART_VERSION")

if [[ $APPLY -eq 0 ]]; then
  log "DRY RUN — server-side rendering the chart (no changes will be made)"
  helm "${HELM_ARGS[@]}" --dry-run=server
  echo
  warn "This was a dry run. Re-run with --apply to install on the cluster."
  if [[ $WITH_QUEUES -eq 1 ]]; then
    warn "--with-queues objects (ResourceFlavor/ClusterQueue) are printed below but NOT applied."
    render_queues
  fi
  exit 0
fi

log "Installing/upgrading $CHART as release '$RELEASE' in $OPERATOR_NS"
helm "${HELM_ARGS[@]}"

log "Waiting for the Kueue controller to become available"
kubectl "${KCTL_FLAGS[@]}" -n "$OPERATOR_NS" rollout status \
  deploy/kueue-controller-manager --timeout=180s

log "Topologies on the cluster:"
kubectl "${KCTL_FLAGS[@]}" get topologies.kueue.x-k8s.io 2>/dev/null || \
  warn "no Topology CRs found yet (the chart may still be reconciling)"

if [[ $WITH_QUEUES -eq 1 ]]; then
  log "Applying ResourceFlavor + ClusterQueue ($CLUSTER_QUEUE)"
  render_queues | kubectl "${KCTL_FLAGS[@]}" apply -f -
fi

log "Done. Point the Iris cluster config at the ClusterQueue; Iris creates the"
log "LocalQueue in its namespace at controller start:"
cat <<EOF
  kubernetes_provider:
    kueue:
      local_queue: iris-lq            # Iris creates this LocalQueue
      cluster_queue: ${CLUSTER_QUEUE}     # binds to the admin ClusterQueue above
EOF
