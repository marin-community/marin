import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: { index: './src/main.ts' },
  },
  output: {
    distPath: { root: 'dist' },
    assetPrefix: 'auto',
    inlineScripts: true,
    inlineStyles: true,
  },
  performance: {
    chunkSplit: { strategy: 'all-in-one' },
  },
  html: {
    template: './src/template.html',
    inject: 'body',
  },
  server: {
    proxy: {
      '/zephyr.dashboard.v1.CoordinatorDashboardService': 'http://localhost:8000',
    },
  },
})
