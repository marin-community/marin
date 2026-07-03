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
    // Vendor speedscope's prebuilt SPA from its npm package into the dashboard's
    // static assets so the controller serves it at /static/speedscope/ for
    // one-click profile viewing. Sourcing it from node_modules keeps the ~1MB
    // bundle out of the repo — `npm ci` fetches it at build time instead.
    copy: [{ from: 'node_modules/speedscope/dist/release', to: 'static/speedscope' }],
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
      '/iris.cluster.WorkerService': 'http://localhost:8081',
      '/proxy': 'http://localhost:8080',
      '/bundles': 'http://localhost:8080',
      '/blobs': 'http://localhost:8080',
      '/health': 'http://localhost:8080',
      '/auth': 'http://localhost:8080',
      '/api': 'http://localhost:8080',
    },
  },
})
