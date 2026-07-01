/// <reference types="@rsbuild/core/types" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// plotly.js-dist-min ships no types; we use a thin local wrapper (utils/plot.ts).
declare module 'plotly.js-dist-min'
