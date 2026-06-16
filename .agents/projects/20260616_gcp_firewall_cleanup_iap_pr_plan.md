# GCP Firewall Cleanup and IAP SSH PR Plan

## Context

During the June 2026 Marin ops rotation, a read-only audit of `hai-gcp-models` found multiple public ingress firewall rules on the default network. The highest-risk pattern is broad source ranges with no target tags or target service accounts, which apply to every GCE VM in the default network.

Current broad-ingress findings:

| Rule | Source | Allow | Targeting | Current impact |
| --- | --- | --- | --- | --- |
| `default-allow-ssh` | `0.0.0.0/0` | `tcp:22` | none | applies to all default-network GCE VMs |
| `default-allow-rdp` | `0.0.0.0/0` | `tcp:3389` | none | applies to all default-network GCE VMs |
| `allow-vllm-8000` | `0.0.0.0/0` | `tcp:8000` | none | applies to all default-network GCE VMs |
| `marin-cc-index` | `0.0.0.0/0` | `tcp:8080` | none | applies to all default-network GCE VMs |
| `default-allow-icmp` | `0.0.0.0/0` | `icmp` | none | applies to all default-network GCE VMs |
| `ivanzhou-tcp-22` | `0.0.0.0/0` | `tcp:22` | `ivanzhou-deployment` | no matching GCE instances in the audit |
| `port-5000` | `0.0.0.0/0` | `tcp:5000` | `port-5000` | no matching GCE instances in the audit |
| `port-8001` | `0.0.0.0/0` | `tcp:8001` | `port-8001` | no matching GCE instances in the audit |
| `allow-openvpn` | `0.0.0.0/0` | `udp:1194` | `openvpn-server` | no matching GCE instances in the audit |

The audit observed 32 GCE instances in the default network; 27 were running with external NAT IPs. TPU healthcheck and TPU pod rules were numerous but scoped by Google healthcheck source ranges or source/target tags and are not the cleanup target.

Important repo mismatch: `lib/iris/OPS.md` already documents connecting to the Iris controller through IAP SSH, but `lib/iris/src/iris/cluster/backends/gcp/controller.py` currently builds `gcloud compute ssh` without `--tunnel-through-iap`; a dry run resolves `iris-controller-marin` to its public IP. Removing `default-allow-ssh` before fixing this would break normal Iris controller access.

## Goal

Move Marin GCP SSH access to IAP first, then remove or re-scope broad public ingress rules with a staged, reversible process.

## Non-Goals

- Do not mutate live GCP firewall state in the code PR.
- Do not delete or disable public SSH until IAP has been validated for all required access paths.
- Do not attempt the larger no-external-IP / Cloud NAT migration in this PR.
- Do not change TPU pod or healthcheck firewall behavior.

## Proposed PR Scope

### 1. Add explicit IAP SSH support in Iris

Add an explicit SSH config option for GCP SSH command construction: `SshConfig.tunnel_through_iap`. Use protobuf field `8`; fields `5` and `6` are reserved and field `7` is already `impersonate_service_account`.

Expected behavior:

- When enabled, `gcloud compute ssh` commands include `--tunnel-through-iap`.
- When disabled, legacy public-IP SSH behavior remains available as a break-glass fallback while `default-allow-ssh` still exists.
- The initial code PR should add support and tests without flipping live cluster config defaults, unless Phase 0 IAP firewall and IAM have already been applied. This keeps PR CI from depending on an unverified live IAP setup.
- A follow-up config flip can set `marin.yaml`, `marin-dev.yaml`, and `smoke-gcp.yaml` to IAP SSH after Phase 0 is complete.
- These configs already set `defaults.ssh.impersonate_service_account: iris-controller@hai-gcp-models.iam.gserviceaccount.com`; the default flip should preserve that impersonation behavior and add IAP on top of it.
- Public-IP fallback is an explicit operator override, not an automatic retry. If an operator sets `tunnel_through_iap: false`, the command emits a visible warning.

PR A surfaces to update:

- `lib/iris/src/iris/rpc/config.proto`
- checked-in generated `config_pb2.py`
- `lib/iris/src/iris/cluster/config.py`
- `lib/iris/src/iris/cluster/backends/gcp/controller.py`
- `lib/iris/src/iris/cluster/backends/gcp/ssh.py`
- `lib/iris/src/iris/cluster/backends/remote_exec.py`
- existing GCP backend tests in `lib/iris/tests/cluster/backends/gcp/test_platform.py`
- `lib/iris/OPS.md`

PR B surfaces to update after Phase 0 live IAP/IAM preflight:

- `lib/iris/config/marin.yaml`
- `lib/iris/config/marin-dev.yaml`
- `lib/iris/config/smoke-gcp.yaml`
- `.github/workflows/iris-smoke-gcp.yaml`

Implementation note: `rigging.tunnel.GcpSshForwardTarget` already has `tunnel_through_iap` and impersonation support for local forwards. The Iris PR should either reuse that path or deliberately keep the Iris-specific command builder while avoiding another inconsistent implementation.

### 2. Resolve GCE and TPU SSH paths before public SSH removal

The controller tunnel uses `gcloud compute ssh`. GCE worker debug uses the GCE remote-exec path. TPU worker debug uses `gcloud compute tpus tpu-vm ssh`, which is a different command path and may not support the same IAP flags in exactly the same way.

The PR should:

- Wire `tunnel_through_iap` through controller SSH.
- Wire `tunnel_through_iap` through GCE remote exec.
- Probe and test whether `gcloud compute tpus tpu-vm ssh` supports the required IAP behavior.
- If TPU IAP SSH cannot be made safe in this PR, explicitly document that `default-allow-ssh` removal is blocked until TPU debug access is resolved.
- PR A wires the opt-in flag into TPU remote exec but does not prove the live TPU command path works with IAP. Public SSH removal remains blocked until a real TPU worker IAP SSH smoke check passes.

### 3. Add an ops firewall cleanup runbook

Document the staged live cleanup procedure and rollback plan. The runbook should be explicit that the PR only enables safe validation; live firewall changes are separate operator actions.

The runbook should include:

- Current audit commands.
- Required preflight checks.
- Disable-first cleanup sequence.
- Validation checks after each stage.
- Executable rollback commands.
- Break-glass recovery if IAP is misconfigured after public SSH is removed.

### 4. Optional read-only audit script

Add a read-only helper script only if it saves operator time without becoming another source of truth. The script may summarize broad-ingress rules, affected target tags, and currently matching GCE instances. It must not mutate firewall state.

## Live Rollout Plan

### Phase 0: Additive IAP enablement and break-glass preflight

These changes are additive and should be done before any access-removing firewall changes:

1. Verify or create an IAP SSH ingress rule. Use `describe` first so an existing rule is not treated as an error:

   ```bash
   gcloud compute firewall-rules describe allow-iap-ssh \
     --project=hai-gcp-models >/dev/null 2>&1 || \
   gcloud compute firewall-rules create allow-iap-ssh \
       --project=hai-gcp-models \
       --network=default \
       --direction=INGRESS \
       --priority=1000 \
       --action=ALLOW \
       --rules=tcp:22 \
       --source-ranges=35.235.240.0/20
   ```

2. Verify IAM for the impersonated Iris SSH service account. Production `marin.yaml`, development `marin-dev.yaml`, and CI `smoke-gcp.yaml` already SSH with `--impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com`. With `--tunnel-through-iap`, IAP authorizes that impersonated service account, so it must have IAP tunnel access and OS Login permissions on the controller and worker VMs before public SSH can be removed:

   ```bash
   gcloud projects get-iam-policy hai-gcp-models \
     --flatten='bindings[].members' \
     --filter='bindings.members:serviceAccount:iris-controller@hai-gcp-models.iam.gserviceaccount.com AND bindings.role:(roles/iap.tunnelResourceAccessor OR roles/compute.osLogin OR roles/compute.osAdminLogin)'
   ```

   Also check instance-level IAM on the controller and at least one representative worker VM before treating Phase 0 as green:

   ```bash
   gcloud compute instances get-iam-policy iris-controller-marin \
     --project=hai-gcp-models \
     --zone=us-central1-a \
     --flatten='bindings[].members' \
     --filter='bindings.members:serviceAccount:iris-controller@hai-gcp-models.iam.gserviceaccount.com AND bindings.role:(roles/iap.tunnelResourceAccessor OR roles/compute.osLogin OR roles/compute.osAdminLogin)'
   ```

3. Verify IAM for human operators who launch Iris commands. Operators do not need to be the IAP SSH principal when impersonation is used, but they do need permission to impersonate the Iris SSH service account. Check both project-level and service-account-level bindings; do not grant broader project-level impersonation if the service-account-level binding is sufficient:

   ```bash
   gcloud projects get-iam-policy hai-gcp-models \
     --flatten='bindings[].members' \
     --filter='bindings.role:roles/iam.serviceAccountTokenCreator AND bindings.members:user:USER_EMAIL'

   gcloud iam service-accounts get-iam-policy \
     iris-controller@hai-gcp-models.iam.gserviceaccount.com \
     --project=hai-gcp-models \
     --flatten='bindings[].members' \
     --filter='bindings.role:roles/iam.serviceAccountTokenCreator AND bindings.members:user:USER_EMAIL'
   ```

   If any cluster config or manual access path does not use service-account impersonation, separately validate the human identity for `roles/iap.tunnelResourceAccessor` and OS Login.

4. Verify IAM for any other impersonated service accounts used by CI or Iris configs, especially `smoke-gcp.yaml`. IAP authorizes the impersonated identity for CI-driven SSH.

5. Verify or create a separately-permissioned break-glass admin identity that can mutate firewall rules even if normal IAP SSH is broken.

6. Verify whether serial console access and IAM are enabled. If they are not enabled and documented, do not list serial console as a reliable rollback path.

7. Validate a real IAP tunnel to `iris-controller-marin`, not only command construction:

   ```bash
   gcloud compute ssh iris-controller-marin \
     --project=hai-gcp-models \
     --zone=us-central1-a \
     --tunnel-through-iap \
     -- -L 10000:localhost:10000 -N
   ```

Phase 0 is a merge gate for any PR that flips `smoke-gcp.yaml` or other live cluster defaults to `tunnel_through_iap: true`, because `.github/workflows/iris-smoke-gcp.yaml` runs on pull requests that touch Iris code. The safest sequence is:

1. PR A: add the `tunnel_through_iap` config, command construction, tests, and runbook while leaving live config defaults unchanged.
2. Apply Phase 0 live additive firewall/IAM changes.
3. PR B: flip `marin.yaml`, `marin-dev.yaml`, and `smoke-gcp.yaml` defaults to IAP and validate CI.

If choosing a single PR instead, open it as a draft and do not un-draft or merge it until Phase 0 is complete and the smoke workflow has passed with IAP enabled.

### Phase 1: Merge and soak Iris IAP defaults

After the PR lands:

1. Verify `iris --cluster=marin ...` opens the controller tunnel through IAP.
2. Verify storage report, egress report, ferry/canary workflows, and normal `iris job` submission.
3. Verify `.github/workflows/iris-smoke-gcp.yaml` still bootstraps its ephemeral controller through IAP. Firewall rules are VPC-global, so the single `allow-iap-ssh` rule covers the smoke workflow's region as long as IAM is correct.
4. Verify GCE worker debug access through `GceRemoteExec`.
5. Verify TPU worker debug access through `GcloudRemoteExec`, including CLI flag support, OS Login metadata, and IAP-range firewall reachability, or explicitly gate public SSH removal until TPU access is understood.
6. Soak for 24 to 72 hours while public SSH remains enabled. Public fallback should warn loudly if used.

### Phase 2: Disable stale tagged rules

Disable, observe, then delete only after a soak:

```bash
for rule in ivanzhou-tcp-22 port-5000 port-8001 allow-openvpn; do
  gcloud compute firewall-rules update "$rule" \
    --project=hai-gcp-models \
    --disabled
done
```

Pre-check: confirm there are no instance templates, managed instance groups, or launch paths that still rely on these tags.

### Phase 3: Disable clearly bad no-target application rules

Before disabling, use Firewall Insights, VPC flow logs, or service-owner confirmation to determine whether the ports have recent legitimate traffic:

- `allow-vllm-8000`
- `marin-cc-index`

If still needed, re-scope rather than delete:

- Add explicit target tags or target service accounts.
- Restrict source ranges to Stanford/VPN/IAP/proxy where possible.
- Remove the no-target broad rule only after the scoped replacement is validated.

### Phase 4: Disable RDP

Disable `default-allow-rdp` unless a current Windows/RDP dependency is identified:

```bash
gcloud compute firewall-rules update default-allow-rdp \
  --project=hai-gcp-models \
  --disabled
```

### Phase 5: Public SSH removal

Only after IAP has soaked and all access paths are validated:

```bash
gcloud compute firewall-rules update default-allow-ssh \
  --project=hai-gcp-models \
  --disabled
```

Keep it disabled for another soak window before deletion.

### Phase 6: ICMP decision

Make an explicit decision on `default-allow-icmp`. Public ICMP is less severe than SSH/RDP/app ports, but it still exposes instance reachability. Either retain it deliberately or restrict/disable it after the higher-risk cleanup is complete.

## Rollback Plan

Before any live firewall mutation, snapshot all target rules and also save explicit recreate commands for each rule. A JSON `describe` snapshot is useful for review, but it cannot be fed directly to `gcloud compute firewall-rules create` during an incident.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
SNAP=/tmp/marin-firewall-snapshot-$TS
mkdir -p "$SNAP"
for rule in \
  default-allow-ssh default-allow-rdp default-allow-icmp \
  allow-vllm-8000 marin-cc-index \
  ivanzhou-tcp-22 port-5000 port-8001 allow-openvpn; do
  gcloud compute firewall-rules describe "$rule" \
    --project=hai-gcp-models \
    --format=json > "$SNAP/$rule.json"
done
```

At minimum, write the public-SSH break-glass recreate command next to the snapshot and verify it matches the live rule priority before disabling or deleting the rule:

```bash
cat > "$SNAP/recreate-default-allow-ssh.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
gcloud compute firewall-rules create default-allow-ssh \
  --project=hai-gcp-models \
  --network=default \
  --direction=INGRESS \
  --priority=65534 \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=0.0.0.0/0
EOF
chmod +x "$SNAP/recreate-default-allow-ssh.sh"
```

For non-break-glass rules, generate or hand-write equivalent recreate commands before deletion. Do not delete a rule if its recreate command is not executable.

Rollback from a disabled rule:

```bash
gcloud compute firewall-rules update RULE_NAME \
  --project=hai-gcp-models \
  --no-disabled
```

Rollback from deleted rule:

- Recreate from the saved rule-specific `gcloud compute firewall-rules create ...` command, such as `"$SNAP/recreate-default-allow-ssh.sh"` for public SSH.
- Prefer restoring only the specific rule required rather than reverting the entire firewall cleanup.

Rollback from Iris IAP default:

- Manually set `tunnel_through_iap: false` in the relevant cluster config or use an explicit CLI/config override. Iris should not automatically retry public-IP SSH after IAP failure.
- This rollback only works while `default-allow-ssh` still exists. Once public SSH is disabled or deleted, the only valid rollback is fixing IAP or re-enabling/recreating public SSH.

Break-glass if locked out after public SSH removal:

- Use the pre-verified GCP admin identity to re-enable or recreate `default-allow-ssh` from the runnable recreate command.
- If SSH is still impossible, use serial console only if serial console access and IAM were verified in Phase 0.
- Do not rely on `tunnel_through_iap: false` after public SSH is removed.

Rollback triggers:

- `iris --cluster=marin` cannot open controller tunnel.
- Scheduled storage or egress workflows fail due to SSH/IAP access.
- `iris-smoke-gcp` fails due to controller bootstrap or SSH/IAP access.
- GCE worker debug access fails and no replacement path exists.
- TPU worker debug access fails and public SSH removal was not explicitly gated.
- A known service owner reports breakage on port `8000` or `8080`.
- Firewall Insights or flow logs show legitimate traffic blocked by a disabled rule.

## Required Tests

Unit tests:

- GCP controller SSH command includes `--tunnel-through-iap` when enabled.
- GCP controller SSH command omits `--tunnel-through-iap` when disabled.
- `GceRemoteExec` commands include `--tunnel-through-iap` when enabled.
- `GcloudRemoteExec` TPU commands include `--tunnel-through-iap` when enabled, if the TPU command supports IAP.
- TPU validation covers OS Login metadata and firewall reachability, not only command-line flag construction.
- Remote exec preserves impersonation flags and SSH key behavior.
- A test fixture config can resolve to IAP SSH when `tunnel_through_iap: true`.
- In the follow-up config-flip PR, `marin.yaml`, `marin-dev.yaml`, and `smoke-gcp.yaml` resolve to IAP SSH defaults.
- Non-GCP/manual providers are unaffected.
- Explicit public fallback emits a visible warning and is never an automatic retry after IAP failure.

Integration/smoke checks before public SSH removal:

- Real IAP tunnel to `iris-controller-marin`.
- `iris --cluster=marin job list` or another harmless controller RPC through the named cluster.
- One lightweight GCE worker debug command through `GceRemoteExec`, if available.
- One TPU worker debug access check, or an explicit note that public SSH removal is blocked until TPU access is validated.
- Manual `iris-smoke-gcp` workflow run.
- `ops-egress-report` and `ops-storage-report` dry-run/manual runs if they rely on the same SSH path.

## Draft PR Description

Title: `Add IAP SSH support for Marin GCP cluster access`

Summary:

- Adds explicit IAP SSH support to GCP SSH command construction.
- Preserves existing service-account impersonation and makes IAP opt-in for a safe first PR.
- Adds tests for IAP and public-IP SSH command construction.
- Documents the staged firewall cleanup and rollback plan needed before broad public ingress rules can be disabled.

Test plan:

- `uv run pytest lib/iris/tests/cluster/backends/gcp/test_platform.py`
- `uv run pyrefly`
- `./infra/pre-commit.py --review`
- Manual IAP tunnel validation before any live firewall cleanup.

## CC Reviews

- Approach review: `70c19520-9f3f-4eb1-a530-3c3d5a99b20b`, `claude-opus-4-8`, `ctc`, `env -u ANTHROPIC_API_KEY`.
- Draft plan review: `049793a5-0467-45f4-8757-1a233931a8e5`, `claude-opus-4-8`, `ctc`, `env -u ANTHROPIC_API_KEY`.
