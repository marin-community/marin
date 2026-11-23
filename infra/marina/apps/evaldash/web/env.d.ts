/// <reference types="@rsbuild/core/types" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// The shared transcript renderer is plain JavaScript with a global fallback for no-build
// hosts; this keeps the bundler import typed.
declare module '@marina/transcript.js' {
  const TranscriptView: {
    fenceSegments(text: string): { kind: 'prose' | 'code'; text: string }[]
    normalizeTurns(source: unknown): Record<string, unknown>[]
    renderTranscript(container: HTMLElement, source: unknown, options?: Record<string, unknown>): void
    renderStepCard(container: HTMLElement, step: object, options?: Record<string, unknown>): void
    renderFenced(container: HTMLElement, text: string): void
    renderMessageBody(container: HTMLElement, text: string, previewChars?: number | null): void
  }
  export default TranscriptView
}
