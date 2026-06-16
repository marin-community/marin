# Ops Log

## 2026-06-16: Broad GCP Firewall Ingress Audit

Read-only audit of `hai-gcp-models` found several broad public ingress firewall rules on the default network. The highest-risk rules have source `0.0.0.0/0` and no target tags/service accounts, so they apply to every GCE VM in the default network:

- `default-allow-ssh` allows `tcp:22`.
- `default-allow-rdp` allows `tcp:3389`.
- `allow-vllm-8000` allows `tcp:8000`.
- `marin-cc-index` allows `tcp:8080`.
- `default-allow-icmp` allows public ICMP.

The audit also found stale-looking broad tagged rules with no matching GCE instances: `ivanzhou-tcp-22`, `port-5000`, `port-8001`, and `allow-openvpn`.

Operational finding: `lib/iris/OPS.md` documents IAP SSH, but the current Iris GCP SSH command path does not pass `--tunnel-through-iap`; public SSH is still required for normal controller tunneling. Cleanup should therefore be staged: add/validate IAP SSH first, then disable broad public ingress rules with snapshots and rollback.

Follow-up CC review emphasized five concrete rollout requirements before public SSH removal:

- Validate the actual SSH principal. Current Iris configs already impersonate `iris-controller@hai-gcp-models.iam.gserviceaccount.com`, so the service account needs IAP and OS Login, and human operators need service-account impersonation permission.
- Include `smoke-gcp.yaml` and the `iris-smoke-gcp` workflow in the IAP migration.
- Treat TPU worker debug SSH as a blocking access path unless IAP is wired and tested there too.
- Save executable firewall recreate commands, not only JSON snapshots.
- Pre-verify break-glass admin access and serial console assumptions before disabling any rule.
- Do not flip `smoke-gcp.yaml` to IAP before the live IAP firewall rule and CI service-account IAM are ready, because the Iris smoke workflow runs on Iris pull requests.

Draft PR and rollback plan: `.agents/projects/20260616_gcp_firewall_cleanup_iap_pr_plan.md`.
