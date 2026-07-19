import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: {
      index: './src/main.ts',
    },
  },
  output: {
    distPath: { root: 'dist' },
    // The dashboard ships as one self-contained HTML file, committed at the
    // sibling ../quick_serve_dashboard.html and served via importlib.resources —
    // so every script and stylesheet must inline. scripts/emit.mjs verifies
    // nothing escaped into a separate asset.
    inlineScripts: true,
    inlineStyles: true,
  },
  performance: {
    chunkSplit: { strategy: 'all-in-one' },
  },
  html: {
    template: './src/template.html',
    // Inline scripts cannot defer, so inject at the end of <body> — in <head>
    // the app would mount before #app exists and render a blank page.
    inject: 'body',
  },
})
