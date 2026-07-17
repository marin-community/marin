export type UtcWeekday = 0 | 1 | 2 | 3 | 4 | 5 | 6;
export type NightlyGroup = "marin" | "forks";
export type NightlySubgroup =
  | "training"
  | "data"
  | "cluster"
  | "evaluation"
  | "rl"
  | "inference";

export interface ExpectedDuration {
  minSeconds: number;
  maxSeconds: number;
  provenance: string;
  evidenceUrls?: readonly string[];
}

export interface NightlyLaneConfig {
  id: string;
  label: string;
  shortLabel: string;
  group: NightlyGroup;
  subgroup: NightlySubgroup;
  repository: `marin-community/${string}`;
  workflowFile: string;
  branch: string;
  schedule: {
    weekdays: readonly UtcWeekday[];
    hour: number;
    minute: number;
  };
  activeFrom?: string;
  activeUntil?: string;
  overdueGraceMinutes: number;
  overdueGraceProvenance: string;
  expectedDuration?: ExpectedDuration;
}

const ALL_DAYS: readonly UtcWeekday[] = [0, 1, 2, 3, 4, 5, 6];

export const NIGHTLY_LANES: readonly NightlyLaneConfig[] = [
  {
    id: "tpu-ferry",
    label: "TPU ferry",
    shortLabel: "TPU ferry",
    group: "marin",
    subgroup: "training",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-ferry.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 6, minute: 0 },
    overdueGraceMinutes: 300,
    overdueGraceProvenance: "Observed scheduler delay up to 3h37; 5h tolerance",
    expectedDuration: {
      minSeconds: 60 * 60,
      maxSeconds: 195 * 60,
      provenance: "Verified 61–192m successes; workflow expectation ~2h",
      evidenceUrls: [
        "https://github.com/marin-community/marin/actions/runs/29482562619",
        "https://github.com/marin-community/marin/actions/runs/29185545233",
        "https://github.com/marin-community/marin/blob/main/.github/workflows/marin-canary-ferry.yaml#L107-L116",
      ],
    },
  },
  {
    id: "cw-gpu-ferry",
    label: "CoreWeave GPU ferry",
    shortLabel: "CW ferry",
    group: "marin",
    subgroup: "training",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-ferry-coreweave.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 10, minute: 0 },
    overdueGraceMinutes: 300,
    overdueGraceProvenance: "Observed scheduler delay up to 3h36; 5h tolerance",
    expectedDuration: {
      minSeconds: 15 * 60,
      maxSeconds: 40 * 60,
      provenance: "Verified 17–34m successes; workflow expectation ~35m",
      evidenceUrls: [
        "https://github.com/marin-community/marin/actions/runs/29411190268",
        "https://github.com/marin-community/marin/actions/runs/29328246308",
        "https://github.com/marin-community/marin/blob/main/.github/workflows/marin-canary-ferry-coreweave.yaml#L161-L176",
      ],
    },
  },
  {
    id: "grug-multislice",
    label: "Grug multislice",
    shortLabel: "Grug",
    group: "marin",
    subgroup: "training",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-grug-multislice.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 10, minute: 30 },
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 3h; 4h tolerance",
  },
  {
    id: "datakit-t1",
    label: "Datakit tier 1",
    shortLabel: "Data T1",
    group: "marin",
    subgroup: "data",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-datakit-tier1.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 6, minute: 30 },
    overdueGraceMinutes: 360,
    overdueGraceProvenance: "Observed scheduler delay up to 4h42; 6h tolerance",
    expectedDuration: {
      minSeconds: 65 * 60,
      maxSeconds: 85 * 60,
      provenance: "Verified recent successes spanning 69–81m",
      evidenceUrls: [
        "https://github.com/marin-community/marin/actions/runs/29402548908",
        "https://github.com/marin-community/marin/actions/runs/29566962629",
      ],
    },
  },
  {
    id: "datakit-t2",
    label: "Datakit tier 2",
    shortLabel: "Data T2",
    group: "marin",
    subgroup: "data",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-datakit-tier2.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 7, minute: 0 },
    overdueGraceMinutes: 360,
    overdueGraceProvenance: "Observed scheduler delay up to 4h38; 6h tolerance",
    expectedDuration: {
      minSeconds: 65 * 60,
      maxSeconds: 85 * 60,
      provenance: "Verified recent successes spanning 70–81m",
      evidenceUrls: [
        "https://github.com/marin-community/marin/actions/runs/29320436509",
        "https://github.com/marin-community/marin/actions/runs/29569044292",
      ],
    },
  },
  {
    id: "datakit-t3",
    label: "Datakit tier 3, Mondays",
    shortLabel: "Data T3 · Mon",
    group: "marin",
    subgroup: "data",
    repository: "marin-community/marin",
    workflowFile: "marin-canary-datakit-tier3.yaml",
    branch: "main",
    schedule: { weekdays: [1], hour: 1, minute: 0 },
    overdueGraceMinutes: 480,
    overdueGraceProvenance: "Observed scheduler delay up to 5h15; 8h tolerance",
    expectedDuration: {
      minSeconds: 70 * 60,
      maxSeconds: 180 * 60,
      provenance: "Verified recent successes spanning 73–177m",
      evidenceUrls: [
        "https://github.com/marin-community/marin/actions/runs/29223772199",
        "https://github.com/marin-community/marin/actions/runs/26737698039",
      ],
    },
  },
  {
    id: "cluster-smoke",
    label: "Cluster smoke",
    shortLabel: "Cluster",
    group: "marin",
    subgroup: "cluster",
    repository: "marin-community/marin",
    workflowFile: "marin-cluster-smoke.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 7, minute: 30 },
    activeFrom: "2026-07-17",
    overdueGraceMinutes: 300,
    overdueGraceProvenance: "New lane; 5h initial tolerance aligned with Marin lanes",
  },
  {
    id: "evalchemy",
    label: "Evalchemy",
    shortLabel: "Evalchemy",
    group: "forks",
    subgroup: "evaluation",
    repository: "marin-community/evalchemy",
    workflowFile: "e2e-nightly.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 7, minute: 0 },
    activeFrom: "2026-07-14",
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 2h17; 4h tolerance",
    expectedDuration: {
      minSeconds: 14 * 60,
      maxSeconds: 20 * 60,
      provenance: "Verified recent successes spanning 16–17m",
      evidenceUrls: [
        "https://github.com/marin-community/evalchemy/actions/runs/29486586651",
        "https://github.com/marin-community/evalchemy/actions/runs/29320475861",
      ],
    },
  },
  {
    id: "harbor",
    label: "Harbor",
    shortLabel: "Harbor",
    group: "forks",
    subgroup: "evaluation",
    repository: "marin-community/harbor",
    workflowFile: "marin-nightly.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 8, minute: 0 },
    activeFrom: "2026-07-15",
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 2h09; 4h tolerance",
    expectedDuration: {
      minSeconds: 6 * 60,
      maxSeconds: 12 * 60,
      provenance: "Verified recent successes spanning 7–8m",
      evidenceUrls: [
        "https://github.com/marin-community/harbor/actions/runs/29406682064",
        "https://github.com/marin-community/harbor/actions/runs/29489742635",
      ],
    },
  },
  {
    id: "marinskyrl",
    label: "MarinSkyRL",
    shortLabel: "SkyRL",
    group: "forks",
    subgroup: "rl",
    repository: "marin-community/MarinSkyRL",
    workflowFile: "marin-nightly.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 9, minute: 0 },
    activeFrom: "2026-07-15",
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 1h58; 4h tolerance",
  },
  {
    id: "vllm-gpu",
    label: "vLLM GPU",
    shortLabel: "vLLM GPU",
    group: "forks",
    subgroup: "inference",
    repository: "marin-community/vllm",
    workflowFile: "marin-nightly.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 10, minute: 0 },
    activeFrom: "2026-07-15",
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 1h39; 4h tolerance",
    expectedDuration: {
      minSeconds: 6 * 60,
      maxSeconds: 15 * 60,
      provenance: "Workflow expectation ~8m; reviewed initial range 6–15m",
      evidenceUrls: [
        "https://github.com/marin-community/vllm/blob/main/.github/workflows/marin-nightly.yaml#L113-L133",
        "https://github.com/marin-community/vllm/actions/runs/29576510987",
      ],
    },
  },
  {
    id: "tpu-inference",
    label: "TPU inference",
    shortLabel: "TPU infer",
    group: "forks",
    subgroup: "inference",
    repository: "marin-community/tpu-inference",
    workflowFile: "marin-e2e-nightly.yaml",
    branch: "main",
    schedule: { weekdays: ALL_DAYS, hour: 11, minute: 0 },
    activeFrom: "2026-07-15",
    overdueGraceMinutes: 240,
    overdueGraceProvenance: "Observed scheduler delay up to 1h13; 4h tolerance",
    expectedDuration: {
      minSeconds: 5 * 60,
      maxSeconds: 10 * 60,
      provenance: "Verified recent successes spanning 6.5–6.8m",
      evidenceUrls: [
        "https://github.com/marin-community/tpu-inference/actions/runs/29414101949",
        "https://github.com/marin-community/tpu-inference/actions/runs/29497271847",
      ],
    },
  },
] as const;

export function workflowSourceUrl(lane: NightlyLaneConfig): string {
  return `https://github.com/${lane.repository}/blob/${lane.branch}/.github/workflows/${lane.workflowFile}`;
}

export function validateNightlyLanes(lanes: readonly NightlyLaneConfig[]): void {
  const ids = new Set<string>();
  const workflows = new Set<string>();
  for (const lane of lanes) {
    if (ids.has(lane.id)) throw new Error(`duplicate nightly lane id: ${lane.id}`);
    ids.add(lane.id);

    const workflowKey = `${lane.repository}/${lane.workflowFile}`;
    if (workflows.has(workflowKey)) {
      throw new Error(`duplicate nightly workflow: ${workflowKey}`);
    }
    workflows.add(workflowKey);

    if (lane.schedule.weekdays.length === 0) {
      throw new Error(`${lane.id}: schedule must include at least one weekday`);
    }
    if (lane.schedule.hour < 0 || lane.schedule.hour > 23) {
      throw new Error(`${lane.id}: invalid UTC hour`);
    }
    if (lane.schedule.minute < 0 || lane.schedule.minute > 59) {
      throw new Error(`${lane.id}: invalid UTC minute`);
    }
    if (lane.overdueGraceMinutes < 0) {
      throw new Error(`${lane.id}: grace must be non-negative`);
    }
    if (lane.activeFrom && lane.activeUntil && lane.activeFrom > lane.activeUntil) {
      throw new Error(`${lane.id}: activeUntil precedes activeFrom`);
    }
    if (
      lane.expectedDuration &&
      (lane.expectedDuration.minSeconds < 0 ||
        lane.expectedDuration.minSeconds > lane.expectedDuration.maxSeconds)
    ) {
      throw new Error(`${lane.id}: invalid expected duration range`);
    }
  }
}

validateNightlyLanes(NIGHTLY_LANES);
