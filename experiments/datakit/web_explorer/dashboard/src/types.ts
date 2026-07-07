/** Payload of `GET api/overview` (see server.py::build_app.overview). */

export interface Bucket {
  cluster_id: number
  quality_bucket: number
  quality_range: string
  total_elements: number
  total_tokens: number
}

export interface SourceSummaryRow {
  source: string
  docs_est: number | null
  q_avg: number | null
  q_sd: number | null
  q_zero: number | null
  drop_rate: number | null
  dup_prevalence: number | null
  dup_largest: number | null
  dup_avg_size: number | null
  decon_pct: number | null
}

export interface Overview {
  store_path: string
  data_prefix: string
  cluster_view: number
  quality_thresholds: number[]
  n_quality_buckets: number
  tokenizer: string
  verified: boolean
  sources: string[]
  resolved: {
    normalize: string[]
    decontam: string[]
    cluster_assign: string[]
    quality: string[]
  }
  dedup: string | null
  counters: Record<string, number>
  buckets: Bucket[]
  source_summary: SourceSummaryRow[]
  /** false while the app is still counting per-source sizes in the background. */
  source_summary_ready: boolean
  source_summary_total: number
}

export interface HistBucket {
  lo: number
  hi: number
  docs: number
}

export interface DedupCluster {
  cluster_id: string | number
  sampled_size: number
  members: { canonical: boolean; text: string }[]
}
