# marin.community/belay

A Cloudflare Worker that serves the public Marin Belay page at `https://marin.community/belay/` with a clean
address bar. The page is a public Marina applet, which Marina serves only by id
(`https://public.applets.marina.oa.dev/a/33801d46-e280-46f9-b994-2c78603cdf15/`); the Worker, routed on
`marin.community/belay*`, fetches each request from there and returns it under `/belay/`. The page and its API
calls use paths relative to the address bar, so the Worker only rewrites Marina's redirects (the root to its
versioned path). It forwards GET and HEAD only.

The `marin.community` zone is not managed by this repository's Pulumi stacks, so deploy the Worker with an
account that can edit that zone:

```sh
cd infra/belay-proxy
npx wrangler deploy
```

The route in `wrangler.toml` attaches it to `marin.community/belay*`; every other path on `marin.community` still
goes to the docs site. To check it locally against the live page, import `worker.js` in Node and call its
`fetch` with a `Request` for `https://marin.community/belay/`.
