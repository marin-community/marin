/// <reference types="@rsbuild/core/types" />

declare module '@fontsource-variable/*'

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
