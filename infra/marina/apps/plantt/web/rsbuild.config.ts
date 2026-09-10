import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  html: { template: './index.html' },
  resolve: {
    alias: { vue: './node_modules/vue', '@marina': '../../../web' },
  },
  source: {
    entry: { index: './src/main.js' },
  },
  output: {
    distPath: { root: '../dist' },
    assetPrefix: '/plantt/',
    cleanDistPath: true,
  },
  server: {
    port: Number(process.env.PORT ?? 3000),
  },
})
