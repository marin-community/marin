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
  },
  html: {
    template: './src/template.html',
    templateParameters: { title: 'Echo · Marin' },
  },
  server: {
    proxy: {
      '/api': DEV_API_ORIGIN,
    },
  },
})
