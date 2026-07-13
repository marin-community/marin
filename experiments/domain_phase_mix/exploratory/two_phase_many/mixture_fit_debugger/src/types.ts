export type ModelId =
  | "canonical"
  | "effective_exposure"
  | "effective_exposure_geometry"
  | "separate_heads"
  | "grp";
export type ExplorerTab = "mixtures" | "fit";
export type ViewMode = "prediction" | "residual" | "standardized";
export type SortMode = "difference" | "exposure" | "domain";

export interface NoiseReference {
  n: number;
  mean: number;
  standardDeviation: number;
  differenceStandardDeviation: number;
}

export interface TargetMetadata {
  id: string;
  label: string;
  metricColumn: string;
  lowerIsBetter: boolean;
  noiseReference: NoiseReference;
  noiseLabel?: string;
}

export interface DomainMetadata {
  id: string;
  label: string;
  group: string;
  proportionalWeight: number;
  tokenCount: number;
  phase0EpochFactor: number;
  phase1EpochFactor: number;
}

export interface RowDiagnostics {
  phaseTv: number;
  aggregateTvToProportional: number;
  aggregateKlToProportional: number;
  maxEpoch: number;
  nearestFitId: string;
  supportDistance: number;
}

export interface MixtureRow {
  id: string;
  name: string;
  split: "fit" | "heldout" | "noise_reference" | "candidate";
  policyFamily: string | null;
  phaseFamily: string | null;
  phaseStructure: string | null;
  panel: string | null;
  method: string | null;
  sourceExperiment: string | null;
  wandbUrl: string | null;
  interventionType: string | null;
  targetDomain: string | null;
  directionType: string | null;
  directionId: string | null;
  isSharedAlias: boolean;
  pairedRow: string | null;
  candidateTarget: string | null;
  observed: Record<string, number | null>;
  phase0: number[];
  phase1: number[];
  aggregate: number[];
  phase0Epochs: number[];
  phase1Epochs: number[];
  totalEpochs: number[];
  diagnostics: RowDiagnostics;
}

export interface MetricSummary {
  n: number;
  rmse: number | null;
  mae: number | null;
  spearman: number | null;
}

export interface ModelDiagnostics {
  fitOof: MetricSummary;
  heldout: MetricSummary;
  heldoutSinglePhase: MetricSummary;
  heldoutTwoPhase: MetricSummary;
}

export interface PredictionSeries {
  prediction: Array<number | null>;
  fullFitPrediction: Array<number | null>;
}

export interface BaselineOption {
  id: string;
  label: string;
}

export interface ModelMetadata {
  id: ModelId;
  label: string;
  description: string;
}

export interface FitParameter {
  key: string;
  symbol: string;
  value: number | null;
  role: string;
  scope: "global" | "domain" | "group" | string;
  domainId: string | null;
  groupLabel: string | null;
  transformedValue: number | null;
  transformedLabel: string | null;
  unit: string | null;
}

export interface FitDetail {
  modelId: ModelId;
  modelLabel: string;
  description: string;
  parameterCount: number;
  parameters: FitParameter[];
  diagnostics: {
    oof: MetricSummary;
    train: MetricSummary;
  };
  tuning: Record<string, unknown>;
  protocol: string;
  caveats: string[];
}

export interface NikeSwooshDiagnostic {
  sliceDefinition: string;
  xLabel: string;
  yLabel: string;
  observed: { x: number[]; y: number[]; rowIds: string[] };
  grid: number[];
  sliceFit: { label: string; prediction: number[]; minimumX: number; minimumY: number };
  overallFit: { label: string; prediction: number[]; minimumX: number; minimumY: number };
}

export interface SwarmData {
  id: string;
  label: string;
  description: string;
  dataset: {
    label: string;
    fitDesignCount: number;
    rawFitObservationCount: number;
    heldoutCount: number;
    noiseReferenceCount: number;
    supplementalCandidateCount: number;
    phaseFractions: [number, number];
    targetBudget: number;
    oofSeeds: number[];
    fitProtocol: string;
  };
  domains: DomainMetadata[];
  targets: Record<string, TargetMetadata>;
  rows: MixtureRow[];
  predictions: Record<string, Record<ModelId, PredictionSeries>>;
  diagnostics: Record<string, Record<ModelId, ModelDiagnostics>>;
  baselines: Record<string, BaselineOption[]>;
  fits: Record<string, Record<ModelId, FitDetail>>;
  nikeSwoosh: Record<string, Partial<Record<ModelId, NikeSwooshDiagnostic>>>;
  provenance: Record<string, unknown>;
}

export interface DashboardData {
  schemaVersion: 2;
  generatedAt: string;
  models: Record<ModelId, ModelMetadata>;
  swarms: Record<string, SwarmData>;
  provenance: Record<string, unknown>;
}

export interface PointDatum {
  row: MixtureRow;
  rowIndex: number;
  observed: number;
  prediction: number;
  fullFitPrediction: number;
  residual: number;
  standardizedResidual: number;
}

export interface DashboardState {
  swarm: string;
  target: string;
  model: ModelId;
  tab: ExplorerTab;
  view: ViewMode;
  selectedId: string | null;
  baselineId: string;
  showFit: boolean;
  showHeldout: boolean;
  showNoise: boolean;
  hideAliases: boolean;
  phaseFamily: "all" | "single_phase" | "two_phase";
  search: string;
  sort: SortMode;
  parameterDomain: string;
  parameterGroup: string;
}
