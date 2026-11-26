import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  html: { template: './src/template.html', templateParameters: { title: 'Marin Evals' } },
  // `infra/marina/web` holds the tokens and the shell every app imports; `vue` is named
  // because that directory has no `node_modules` of its own.
  resolve: {
    alias: { vue: './node_modules/vue', '@marina': '../../../web' },
  },
  source: {
    entry: { index: './src/main.ts' },
  },
  // The kernel serves this app from `apps/evaldash/dist` under `/evaldash/`, so every asset
  // URL the page asks for carries that prefix. `cleanDistPath` is stated because the
  // directory sits outside this package and rsbuild leaves such a directory alone unless
  // told; without it the hashed assets of every previous build stay in what the kernel serves.
  output: {
    distPath: { root: '../dist' },
    assetPrefix: '/evaldash/',
    cleanDistPath: true,
  },
})
