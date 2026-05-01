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
    assetPrefix: '/',
  },
  html: {
    template: './src/template.html',
    templateParameters: { title: 'Finelog Dashboard' },
  },
  server: {
    proxy: {
      '/finelog.logging.LogService': 'http://localhost:10001',
      '/finelog.stats.StatsService': 'http://localhost:10001',
      '/health': 'http://localhost:10001',
    },
  },
})
