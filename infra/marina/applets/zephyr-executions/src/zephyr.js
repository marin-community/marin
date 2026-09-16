// Pure helpers shared by the stage list, the stage graph, and the reducer table.

export const REDUCER_PAGE_SIZE = 20;
export const EXECUTION_LIMIT = 100;
export const STAGE_NODE_WIDTH = 224;
export const STAGE_NODE_HEIGHT = 96;
const STAGE_COLUMN_GAP = 32;
const STAGE_ROW_GAP = 28;
const STAGE_PADDING = 8;

export const STAGE_VIEW_STORAGE_KEY = "zephyr-executions-stage-view";
export const STAGE_VIEWS = ["List", "Graph"];
export const IRIS_DASHBOARD = "https://iris.oa.dev";

export function stageLayout(stages) {
  const nodes = [];
  const columns = new Map();
  const lanes = new Map();
  let nextLane = 0;
  for (const stage of stages) {
    const column = stage.dependencies.length
      ? Math.max(
          ...stage.dependencies.map((dependency) => {
            const parent = columns.get(dependency);
            if (parent === undefined) throw new Error(`Missing or unordered stage dependency: ${dependency}`);
            return parent + 1;
          }),
        )
      : 0;
    const row = stage.dependencies.length ? lanes.get(stage.dependencies[0]) : nextLane++;
    columns.set(stage.stage_name, column);
    lanes.set(stage.stage_name, row);
    nodes.push({
      ...stage,
      x: column * (STAGE_NODE_WIDTH + STAGE_COLUMN_GAP) + STAGE_PADDING,
      y: row * (STAGE_NODE_HEIGHT + STAGE_ROW_GAP) + STAGE_PADDING,
    });
  }
  const byName = new Map(nodes.map((node) => [node.stage_name, node]));
  const edges = nodes.flatMap((node) =>
    node.dependencies.map((dependency) => {
      const parent = byName.get(dependency);
      return {
        key: `${dependency}:${node.stage_name}`,
        x1: parent.x + STAGE_NODE_WIDTH,
        y1: parent.y + STAGE_NODE_HEIGHT / 2,
        x2: node.x,
        y2: node.y + STAGE_NODE_HEIGHT / 2,
      };
    }),
  );
  return {
    nodes,
    edges,
    width: Math.max(STAGE_NODE_WIDTH + STAGE_PADDING * 2, ...nodes.map((node) => node.x + STAGE_NODE_WIDTH + STAGE_PADDING)),
    height: Math.max(STAGE_NODE_HEIGHT + STAGE_PADDING * 2, ...nodes.map((node) => node.y + STAGE_NODE_HEIGHT + STAGE_PADDING)),
  };
}

export function stageView(stages, saved) {
  if (saved === "List" || saved === "Graph") return saved;
  return stages.some((stage) => stage.stage_name.startsWith("join-right-")) ? "Graph" : "List";
}

export function joinStep(stage, stages) {
  const branch = stage.stage_name.match(/^join-right-(\d+)-(\d+)-stage\d+$/);
  if (!branch) return null;
  const prefix = `join-right-${branch[1]}-${branch[2]}-stage`;
  const chain = stages.filter((item) => item.stage_name.startsWith(prefix));
  const step = chain.findIndex((item) => item.stage_name === stage.stage_name) + 1;
  return { parentStage: Number(branch[1]), step, total: chain.length };
}

export function relativePayload(value, median) {
  if (value === null || value === undefined || median === null || median === undefined) return "—";
  if (median === 0) return "median 0";
  const ratio = value / median;
  return `${ratio.toLocaleString(undefined, { maximumFractionDigits: ratio >= 10 ? 0 : 1 })}×`;
}

export function formatBytes(bytes) {
  if (!bytes || bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return (val >= 100 ? Math.round(val) : val.toFixed(1)) + " " + units[i];
}

export function irisJobUrl(jobId) {
  return `${IRIS_DASHBOARD}/#/job/${encodeURIComponent(jobId)}`;
}

export function irisTaskUrl(jobId) {
  return `${IRIS_DASHBOARD}/#/job/${encodeURIComponent(jobId)}/task/${encodeURIComponent(jobId + "/0")}`;
}

