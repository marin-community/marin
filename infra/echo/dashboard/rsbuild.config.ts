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
    assetPrefix: 'auto',
  },
  html: {
    template: './src/template.html',
    templateParameters: { title: 'Echo · Marin' },
  },
  server: {
    proxy: {
      '/search': 'http://127.0.0.1:8000',
      '/grep': 'http://127.0.0.1:8000',
      '/chunks': 'http://127.0.0.1:8000',
      '/wiki': 'http://127.0.0.1:8000',
      '/work_log': 'http://127.0.0.1:8000',
    },
  },
})
