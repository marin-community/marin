# Applet examples

- `static-html-applet/` is a one-page applet with checked-in `dist/` files and no
  build step.
- `vue-applet/` builds a Vue single-page app with Vite. Its relative asset base
  works below stable and revision URLs.
- `problem-set-applet/` adds a Python backend and an applet-owned Postgres table.

Each frontend loads JavaScript from packaged files. Marina rejects executable
inline scripts because its content security policy sets `script-src 'self'`.
