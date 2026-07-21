import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  source: { entry: { index: './src/main.ts' } },
  output: { distPath: { root: 'dist' }, assetPrefix: 'auto' },
  html: { template: './src/template.html', templateParameters: { title: 'Marin Ops' } },
  server: { proxy: { '/api': 'http://127.0.0.1:8088', '/healthz': 'http://127.0.0.1:8088' } },
})
