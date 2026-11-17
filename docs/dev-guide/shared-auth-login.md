# Google login for shared grants

Google sign-in establishes a Bao entity, which the grant broker maps to a
deployment-owned principal. OpenFGA supplies current permissions through that
principal. Google account names and browser input cannot select a principal.

The client uses Marin's installed-app Google login with PKCE and state
validation, then presents the Google ID token to Bao's JWT auth method.
Bao verifies Google's signature, issuer, audience and the enrolled claims.
The browser callback belongs to the installed-app login. Bao's separate OIDC
browser-login configuration requires its own OAuth client and redirect setup.
See [Google desktop OAuth](https://developers.google.com/identity/protocols/oauth2/native-app)
and [Bao JWT authentication](https://openbao.org/docs/auth/jwt/).

## Discover the account

Run commands from the Marin checkout with the `marin-rigging[iap]` extra
installed. Select an owner-only directory outside the checkout. Set
`LOGIN_EMAIL` to the independently chosen account and `WORKSPACE_DOMAIN` to
its approved Google Workspace domain.

```bash
LOGIN_PRIVATE="$HOME/.local/state/marin-auth-login-trial"
uv run --no-sync python -m infra.auth.login --directory "$LOGIN_PRIVATE" \
  discover --email "$LOGIN_EMAIL" --hosted-domain "$WORKSPACE_DOMAIN" --headless
```

Open the printed Google authorization URL in a browser. For a remote terminal,
the browser eventually redirects to an unreachable `http://localhost/` page.
Copy the full resulting URL from the address bar and paste it into the hidden
terminal prompt. Its `state` and authorization code are credentials. Keep them
in that prompt and out of chat, logs and screenshots. The client rejects a
bare code and any callback with mismatched state.

Omit `--headless` when the command and browser run on the same workstation.
Use `--client-file` to select another reviewed Google installed-app client.
The standard client is already available in Marin.

Discovery verifies the signed token's audience, issuer, expiry, verified email
and Workspace domain. The Google subject stays stable when the account's email
changes. The private directory receives `google.jwt` and
`google-identity.json`, both with mode `0600`. Google refresh tokens stay in
process memory during discovery. The command persists the short-lived ID token.
Google documents stable subjects and Workspace-domain claims in its
[OpenID Connect reference](https://developers.google.com/identity/openid-connect/openid-connect).

## Run a disposable end-to-end trial

Build the current checkout's native Iris extension using the
[Rust source-build workflow](../tutorials/installation.md#rust-crates-dupekit)
before running this trial. The Iris receiver needs the shared-grant verifier
from this branch. Use the repository's shared Cargo cache. Supply local Bao
and OpenFGA executables.
The trial has been exercised with OpenBao 2.7.2 and OpenFGA 1.22.1. Use the
official [OpenBao installation guide](https://openbao.org/docs/install/) and
[OpenFGA releases](https://github.com/openfga/openfga/releases) to obtain binaries
for the host's operating system and architecture.

```bash
uv run --no-sync python -m infra.auth.prove_login \
  --bao /path/to/bao --openfga /path/to/openfga \
  --google-directory "$LOGIN_PRIVATE" \
  --email "$LOGIN_EMAIL" --hosted-domain "$WORKSPACE_DOMAIN"
```

The trial uses the same operator enrollment, login and session-cleanup
functions as the individual commands below. It starts disposable TLS-enabled
Bao and OpenFGA services, maps the verified Google subject to a trial principal,
and starts a local grant broker and native Iris receiver. It checks denial
before nested group membership, successful grant issuance after membership,
Iris sandbox admission and job status, audience replay denial, root-grant
revocation before token expiry, session revocation, and stable identity after
a fresh Bao login. The Iris backend has no workers, so the job remains queued.

The trial copies the verified Google token into its owner-only temporary
directory. Enrollment files and Bao sessions live there. It removes its local
services, trial credentials and temporary state on exit, including after a
failed trial. The original discovery files remain reusable until the Google
ID token expires. Once the experiment is finished, remove `google.jwt` and
`google-identity.json` from `LOGIN_PRIVATE`.

Run the independent signed-provider checks with the same command and only
the two executable arguments. That mode creates a local TLS OIDC issuer and
signs synthetic ID tokens. The real Bao JWT plugin rejects wrong issuer,
audience, subject, email, email-verification state, Workspace domain and expiry,
and a missing role. It also runs the broker and Iris authorization checks.

## Enroll an approved deployment

Select an approved Bao deployment and supply its verified TLS CA through
`BAO_CACERT`. Set `BAO_ADDR` to the private HTTPS API address. Run the existing
auth bootstrap before enrolling users. Bao requires outbound HTTPS access to
Google discovery and signing-key endpoints. Deployment network policy owns
that access.
The [auth deployment guide](../../infra/auth/README.md#initialize-unseal-and-configure)
describes bootstrap. Its [OpenFGA policy procedure](../../infra/auth/README.md#openfga-publication)
owns model publication and relationships.

An operator supplies an ephemeral `BAO_TOKEN` and independently selects the
expected email, Workspace domain, OAuth client and broker principal.
For this trial, the developer and operator run on the same trusted host. The
operator reads `google.jwt` from the discovery directory; it re-verifies the
token using independently selected account and client settings. For separate
hosts, transfer that file through an approved private channel into a directory
owned by the operator with mode `0700`, keeping the file mode `0600`. Never
transfer the operator's Bao token to the developer.

```bash
uv run --no-sync python -m infra.auth.login --directory "$LOGIN_PRIVATE" \
  enroll --email "$LOGIN_EMAIL" --hosted-domain "$WORKSPACE_DOMAIN" \
  --principal "$TRIAL_PRINCIPAL"
```

For a custom installed-app client, supply the same reviewed `--client-file`
to discovery and enrollment. Enrollment re-verifies the Google signature and
account against operator-selected inputs. Editable discovery metadata supplies
no trusted account or audience settings.

The operator command enables the dedicated `google/` JWT mount with no default
role. It creates an exact-subject role, a named Bao entity and an alias keyed
by Google subject plus auth-mount accessor. Repeating enrollment preserves the
entity UUID. Recreating the mount can bind its new accessor to the same named
entity. An alias already bound to another entity causes enrollment to fail.
The entity must be enabled and have no direct Bao policies. Login checks the
effective policy list and revokes sessions with unexpected inherited identity
authority.
See [Bao identity aliases](https://openbao.org/docs/concepts/identity/).

The resulting session can mint its own `marin-auth` identity JWT and revoke
itself. Its hard lifetime is ten minutes. Enrollment writes the reviewed
binding to private `enrollment.json` and prints a public `subjects` mapping.
Add that mapping to the broker's deployment policy and add the selected
principal to its principal configuration. Apply explicitly bounded trial
relationships through the deployment's OpenFGA operator procedure. The
operator command's output contains the entity UUID and principal needed for
these steps. Start with denied access and add the approved nested membership
to establish the authorization transition.
The fields are `broker_policy.subjects` and
`broker_policy.openfga.principals`, illustrated in the
[staging configuration](../../infra/auth/Pulumi.staging.yaml). The
[relationship examples](../../infra/auth/policy/relationships.json) show a
principal joining a team, the team joining `marin-research`, and that group's
access to an action/resource scope. Publish them only to the intended trial
store. A broker policy-bundle update requires draining existing shared
sessions as described in the deployment guide.

## Login and close the session

The caller uses the enrolled Google token while it remains valid.

```bash
uv run --no-sync python -m infra.auth.login --directory "$LOGIN_PRIVATE" \
  login --bao-issuer "$BAO_IDENTITY_ISSUER"
```

`BAO_IDENTITY_ISSUER` must equal the broker's configured Bao identity issuer,
including `/v1/identity/oidc`. Login checks the returned entity UUID, verifies
the minted identity JWT against that issuer and writes private `bao.token`
and `identity.jwt`. The latter authenticates grant creation, revocation and
token exchange. Grant ownership, permissions and delegation limits come from
broker policy. A partially completed login retains `bao.token` so its session
can still be closed.

Record root grant IDs when creating grants. Revoke those roots before closing
the Bao session.

```bash
uv run --no-sync python -m infra.auth.login --directory "$LOGIN_PRIVATE" \
  close-session --auth-url "$AUTH_URL" --grant-id "$ROOT_GRANT_ID"
```

Repeat `--grant-id` for each root created during the session. With no issued
grants, omit `--auth-url` and `--grant-id`. Cleanup verifies each revocation,
revokes the Bao session and removes its local bearer files. A failed
revocation remains visible and preserves the files needed to retry.

Receiving services check current grant authority on each protected operation.
Revoking a root denies that root and its descendants immediately, including
existing service JWTs. Removing an OpenFGA membership edge denies affected
operations at their next current-policy check.

Bao session revocation prevents further identity minting. A previously minted
Bao identity JWT can authenticate until its ten-minute expiry, and other
unrevoked grants retain their own lifetimes and policy checks. Google browser
sign-out and Google refresh-token revocation have their own lifecycle. Session
cleanup covers the named roots and the local Bao session.
