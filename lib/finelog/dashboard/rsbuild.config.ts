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
    // 'auto' makes chunk URLs resolve against the script tag's origin, so the
    // bundle works under either / or a reverse-proxy prefix like
    // /proxy/log-server/. The base is set via <base href> at serve time.
    assetPrefix: 'auto',
  },
  html: {
    template: './src/template.html',
    templateParameters: { title: 'Finelog Dashboard' },
  },
})
