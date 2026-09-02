# The Marina front-end kit

What every Marina app holds in common: the tokens, the class layer, the shell
around the page, a URN, and a client for the APIs on this origin. Import it; do
not copy it. The kit has no `package.json` — an app's own `node_modules`
supplies Vue.

```
tokens.css   the colours, the fonts, the reset, the phone rules
parts.css    the class layer: .scroll .num .chip .pill .facts .record
Shell.vue    the top bar every app renders inside
client.ts    a call to another app's API on this origin
urn.ts       marin:{app}:{kind}:{id}, and the path that shows it
Linked.vue   a run of prose, with the URNs in it as links
markdown.ts  a Markdown subset, parsed to a tree rather than to HTML
Prose.vue    that tree, drawn through `{{ }}` so markup stays text
```

## How an app opts in

Alias `@marina` to this directory from both the bundler and TypeScript, and name
`vue` alongside it because this directory has no `node_modules` of its own.

```ts
// rsbuild.config.ts
resolve: { alias: { vue: './node_modules/vue', '@marina': '../../../web' } }
```

```jsonc
// tsconfig.json
"paths": { "vue": ["./node_modules/vue"], "@marina/*": ["../../../web/*"] },
"include": ["src/**/*.ts", "src/**/*.vue", "rsbuild.config.ts",
            "../../../web/**/*.ts", "../../../web/**/*.vue"]
```

Take the tokens always. Take `parts.css` when the screens are lists and facts
and states; an app with a look of its own would only have to undo it. To look
different, import the tokens and redefine them afterwards, never inside the
dark-mode media query.

```ts
import '@marina/tokens.css'
import '@marina/parts.css'
```

Render everything inside `Shell.vue`, with the app's own navigation in the `nav`
slot:

```vue
<Shell app="tasktrove">
  <template #nav><RouterLink to="/">Sources</RouterLink></template>
  <RouterView />
</Shell>
```

## The kernel contract

One Cloud Run service serves every app from one origin.

- An app is served under `/{app}/`. Any path under it that is not a file in
  `apps/{app}/dist/` answers with `apps/{app}/dist/index.html`, and files under
  `dist/` are served verbatim. So the bundler's asset prefix is `/{app}/` and
  the router is `createWebHistory('/{app}/')`.
- `GET /api/marina/apps` answers
  `{"apps": [{"name", "title", "description", "path"}, …]}`. `Shell.vue` reads
  it for the switcher.
- `GET /api/marina/me` answers `{"user", "role"}`. `user` is `"anonymous"` in
  local dev, and the shell shows no identity chip for it.
- The kernel sends a Content-Security-Policy per app whose `connect-src` is
  `'self'` plus that app's `connect_src` from its `app.toml`. A fetch to a host
  not on that list is refused by the browser.

`client.ts` posts: `marina.echo.search.query({ q })` is
`POST /api/echo/search/query` with a JSON body, and throws on a non-2xx with the
response text. The two kernel GETs above are plain `fetch`.

## A URN is a link

`marin:{app}:{kind}:{id}` names one row. `path()` in `urn.ts` turns it into
`/{app}/{kind}/{id}` with no lookup, so an app makes its rows linkable by giving
that kind a route. `Linked.vue` renders a paragraph with its URNs as anchors,
both bare and in markdown's `[words](urn)` shape.
