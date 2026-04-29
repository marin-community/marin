import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  source: {
    entry: {
      controller: './src/controller-main.ts',
      worker: './src/worker-main.ts',
    },
  },
  output: {
    distPath: {
      root: 'dist',
    },
    assetPrefix: '/',
  },
  html: {
    template: './src/template.html',
    templateParameters: {
      title: 'Iris Dashboard',
    },
    favicon: './favicon.ico',
  },
  server: {
    proxy: {
      '/iris.cluster.ControllerService': 'http://localhost:8080',
      '/finelog.logging.LogService': 'http://localhost:8080',
      '/iris.cluster.WorkerService': 'http://localhost:8081',
      '/bundles': 'http://localhost:8080',
      '/blobs': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/auth': 'http://localhost:8080',
      '/api': 'http://localhost:8080',
    },
  },
})
