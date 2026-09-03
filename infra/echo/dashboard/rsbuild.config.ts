import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

const DEV_API_ORIGIN = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: {
      index: './src/main.ts',
    },
  },
  output: {
    distPath: { root: 'dist' },
    assetPrefix: 'auto',
    dataUriLimit: { font: 40_000 },
    inlineScripts: true,
    inlineStyles: true,
    legalComments: 'inline',
  },
  // Cloud Run admits one request per Echo instance so CPU-bound searches scale out.
  // Keep the dashboard in one response instead of scaling out for its initial assets.
  performance: {
    chunkSplit: { strategy: 'all-in-one' },
  },
  html: {
    inject: 'body',
    template: './src/template.html',
    templateParameters: { title: 'Echo · Marin' },
  },
  server: {
    proxy: {
      '/api': DEV_API_ORIGIN,
    },
  },
})
