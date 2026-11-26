import { defineConfig } from '@rsbuild/core'
import { pluginVue } from '@rsbuild/plugin-vue'

export default defineConfig({
  plugins: [pluginVue()],
  // `infra/marina/web` holds the tokens and the shell every app renders inside; `vue` is
  // named because that directory has no `node_modules` of its own.
  resolve: {
    alias: { vue: './node_modules/vue', '@marina': '../../../web' },
  },
  source: {
    entry: {
      index: './src/main.ts',
    },
  },
  // The kernel serves this app from `apps/echo/dist` under `/echo/`, so every asset URL
  // the page asks for carries that prefix. `cleanDistPath` is stated because the
  // directory sits outside this package and rsbuild leaves such a directory alone
  // unless told.
  output: {
    distPath: { root: '../dist' },
    assetPrefix: '/echo/',
    cleanDistPath: true,
    dataUriLimit: { font: 40_000 },
  },
  html: {
    inject: 'body',
    template: './src/template.html',
    templateParameters: { title: 'Echo · Marin' },
  },
  server: {
    port: Number(process.env.PORT ?? 3000),
  },
})
