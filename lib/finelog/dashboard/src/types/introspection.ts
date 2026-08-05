/** TypeScript shapes for the server's `/api/*` introspection routes.
 *
 * Manually maintained against `rust/src/server/introspection.rs`.
 */

export interface BuildInfo {
  version: string
  commit: string
  tree: string
  dirty: boolean
  builtAtUnix: number
  rustc: string
  profile: string
}

export interface ProcessInfo {
  pid: number
  hostname: string
  startedAtUnix: number
  uptimeSeconds: number
  rssBytes: number
  vmSizeBytes: number
}

export interface StoreInfo {
  dataDir: string
  remoteLogDir: string
  namespaces: number
  ramBufferBytes: number
  ramChunks: number
}

export interface MetadataCacheInfo {
  limitBytes: number
  sizeBytes: number
  entries: number
  hits: number
}

export interface FormatInfo {
  layoutVersion: number
  targetRowGroupBytes: number
  maxRowGroupRows: number
  sidecarSpanRows: number
}

export interface ServerInfo {
  build: BuildInfo
  process: ProcessInfo
  store: StoreInfo
  metadataCache: MetadataCacheInfo
  format: FormatInfo
}

export interface SidecarInfo {
  columns: string[]
  spans: number
  spanRows: number
}

export interface PhysicalInfo {
  layoutVersion?: number
  layoutCurrent: boolean
  rowGroups: number
  footerBytes: number
  uncompressedBytes: number
  createdBy?: string
  sidecar?: SidecarInfo
}

export interface SegmentInfo {
  path: string
  level: number
  minSeq: number
  maxSeq: number
  rowCount: number
  byteSize: number
  createdAtMs: number
  location: string
  minKeyValue?: string
  maxKeyValue?: string
  physical?: PhysicalInfo
}

export interface SegmentsResponse {
  namespace: string
  segments: SegmentInfo[]
}
