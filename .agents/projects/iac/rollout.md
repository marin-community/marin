# CoreWeave production adoption — rollout plan

Bringing `cw-us-east-02a`, `cw-rno2a`, and `cw-us-east-08a` under real
Pulumi-managed state (`gs://marin-iac-state`), same as `cw-us-west-04a`
(see `gaps.md`).

Order: `cw-us-east-02a` → `cw-rno2a` → `cw-us-east-08a`.

## Plan

1. **Preview each cluster clean**, one at a time.
  ```bash
   cd infra/pulumi
   pulumi login gs://marin-iac-state
   pulumi stack init <cluster> \
     --secrets-provider="gcpkms://projects/hai-gcp-models/locations/us-central1/keyRings/marin-iac-keyring/cryptoKeys/marin-iac-key"
   pulumi config set marin-iac:import true
   pulumi preview
  ```
   Fix any provisioning drift in code (this repo), not on the cluster. No
   NodePool replace/delete is the gate — see README "Adoption check".
  - [x] `cw-us-east-02a` — clean; traefik/cert-manager upgraded live to match pinned versions.
  - [x] `cw-rno2a` — clean; traefik/cert-manager upgraded live to match pinned versions.
  - [x] `cw-us-east-08a` — clean; already on pinned traefik/cert-manager versions.
2. **Cede**: remove `ensure_rbac()`/`ensure_nodepools()` from
  `controller.py`'s `start_controller()`; add `verify_prerequisites()` +
   `PrerequisitesNotProvisionedError` (`spec.md §4`/`§5`).
  - [x] Done.
3. **Push the cede PR.** Do not merge yet.
  - [x] Done.
4. **Real** `pulumi up`, all three clusters, before merging — old Iris is
  still deployed; `force_conflicts` absorbs the ownership fight harmlessly.
  - [ ] `cw-us-east-02a`
  - [ ] `cw-rno2a`
  - [ ] `cw-us-east-08a`
5. **Merge the PR, deploy Iris** to all three clusters (`iris cluster start`).
  - [ ] not started
6. **Final ownership cleanup** — one more `pulumi up` per cluster, in case a
  controller restarted between 4 and 5 under the old code.
  - [ ] `cw-us-east-02a`
  - [ ] `cw-rno2a`
  - [ ] `cw-us-east-08a`
7. **Protect the fleet** — `protect=true`/`retainOnDelete=true` on NodePools.
  - [ ] not started
