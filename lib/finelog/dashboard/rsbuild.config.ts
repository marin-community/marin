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
})
