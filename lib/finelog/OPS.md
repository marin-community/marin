# Finelog Operations

## Access through the Iris IAP endpoint

The Iris controller exposes its finelog server as the `/system/log-server`
endpoint. For `iris.oa.dev`, the public path prefix is
`https://iris.oa.dev/proxy/system.log-server/`.

Authenticate once with the built-in Marin desktop OAuth client (it is
registered as an IAP programmatic client):

```bash
uv run iris --cluster marin login
```

The command caches a refresh token in `~/.config/marin/credentials/marin.json`.
That refresh token mints a short-lived ID token without opening the browser
again:

```bash
IAP_TOKEN="$(uv run python -c 'from rigging.credentials import iap_edge_provider; print(iap_edge_provider("marin").get_token())')"
curl --fail-with-body \
  --header "Proxy-Authorization: Bearer ${IAP_TOKEN}" \
  https://iris.oa.dev/proxy/system.log-server/health
```

IAP consumes `Proxy-Authorization`. Use `Authorization` separately if the
target Iris route also requires an Iris JWT.

The endpoint proxy replaces `/` in endpoint names with `.`. Use
`system.log-server` for `/system/log-server`; `/proxy/system/finelog` addresses
a different endpoint and does not reach finelog.

The finelog CLI uses the same cached credentials when its deployment config
sets `client_url`:

```bash
uv run finelog query marin 'SELECT * FROM "iris.profile" LIMIT 10'
```
