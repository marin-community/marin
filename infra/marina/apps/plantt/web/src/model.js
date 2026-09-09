import LZString from "lz-string";

const { decompressFromEncodedURIComponent } = LZString;

export const PALETTE = ["#8aa0b2", "#ba9f8c", "#8daa91", "#a992b3", "#7fa9aa", "#c3aa72"];
const DAY = 86_400_000;

export const EXAMPLE_PLAN = {
  title: "Example Project Plan",
  note: "Workstreams scheduled across shared compute. Edit the JSON or use the quick-add controls.",
  annotations: [
    { text: "Cluster A online", date: "2026-02-01", target: "@compute", edge: "bottom" },
    { text: "GPU expansion", date: "2026-06-01", target: "@compute", edge: "bottom", color: "#9b6a38" },
  ],
  capacity: [
    { name: "Cluster A", chip: "H100", chips: 128, from: "2026-02-01", color: "#647d92" },
    {
      name: "Cluster B",
      chip: "H100",
      chips: 256,
      from: "2026-04-01",
      color: "#647d92",
      grows: [{ date: "2026-06-01", to: 512 }],
    },
    { name: "TPU pool", chip: "v5p", chips: 256, from: "2026-01-15", to: "2026-05-01", color: "#548a70" },
  ],
  workstreams: [
    {
      name: "Build",
      note: "Core build-out",
      tasks: [
        {
          name: "Foundations",
          start: ["date", "2026-02-03"],
          end: ["weeks", 3],
          cluster: "Cluster A",
          tooltip: "Initial setup on Cluster A.",
        },
        {
          name: "Service v1",
          start: ["after", "Foundations", ["days", 3]],
          end: ["weeks", 6],
          cluster: "Cluster B",
          deps: ["Foundations"],
        },
        {
          name: "Scale-up",
          start: ["date", "2026-06-05"],
          end: ["weeks", 5],
          cluster: "Cluster B",
          deps: ["Service v1"],
        },
      ],
      milestones: [
        { name: "Kickoff", date: "2026-02-01" },
        { name: "Launch", date: "2026-07-20", emoji: "🚀", line: "#c44f43", deps: ["Scale-up"] },
      ],
    },
    {
      name: "Quality",
      note: "Evaluation and hardening",
      tasks: [
        { name: "Eval harness", start: "Service v1", end: ["weeks", 2], cluster: "Cluster A", deps: ["Service v1"] },
        {
          name: "Hardening",
          start: ["after", "Eval harness", ["days", 2]],
          end: ["weeks", 4],
          cluster: "Cluster A",
          deps: ["Eval harness"],
        },
      ],
      milestones: [{ name: "Quality gate", date: "2026-07-01", emoji: "✓", deps: ["Hardening"] }],
    },
    {
      name: "Data",
      note: "Data preparation",
      tasks: [
        { name: "Data prep", start: ["date", "2026-01-20"], end: ["months", 2], cluster: "TPU pool" },
      ],
    },
  ],
};

export function clonePlan(plan) {
  return JSON.parse(JSON.stringify(plan));
}

export function emptyPlan(title = "Untitled plan") {
  return { title, note: "", annotations: [], capacity: [], workstreams: [] };
}

export function parseDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(`Invalid date: '${value}'`);
  }
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.valueOf()) || formatDate(date) !== value) throw new Error(`Invalid date: '${value}'`);
  return date;
}

export function formatDate(date) {
  return date.toISOString().slice(0, 10);
}

export function addDuration(start, duration) {
  if (!Array.isArray(duration) || duration.length !== 2) {
    throw new Error("Task end must be a duration or date tuple");
  }
  const [unit, amount] = duration;
  if (unit === "date") return parseDate(amount);
  if (typeof amount !== "number") throw new Error("Task duration must be numeric");
  const result = new Date(start);
  if (unit === "days") result.setUTCDate(result.getUTCDate() + amount);
  else if (unit === "weeks") result.setUTCDate(result.getUTCDate() + amount * 7);
  else if (unit === "months") result.setUTCMonth(result.getUTCMonth() + amount);
  else throw new Error(`Unknown duration unit: '${unit}'`);
  return result;
}

export function validatePlan(plan) {
  if (!plan || typeof plan !== "object" || Array.isArray(plan)) throw new Error("Chart must be an object");
  if (typeof plan.title !== "string" || !plan.title.trim()) throw new Error("Chart title is required");
  if (!Array.isArray(plan.workstreams)) throw new Error("'workstreams' must be an array");

  const names = new Set();
  const items = [];
  for (const workstream of plan.workstreams) {
    if (!workstream || typeof workstream.name !== "string" || !workstream.name.trim()) {
      throw new Error("Every workstream needs a name");
    }
    if (!Array.isArray(workstream.tasks)) throw new Error(`Workstream '${workstream.name}' needs a tasks array`);
    if (workstream.milestones != null && !Array.isArray(workstream.milestones)) {
      throw new Error(`Workstream '${workstream.name}' milestones must be an array`);
    }
    for (const item of [...workstream.tasks, ...(workstream.milestones || [])]) {
      if (!item || typeof item.name !== "string" || !item.name.trim()) throw new Error("Every item needs a name");
      if (names.has(item.name)) throw new Error(`Duplicate task or milestone name: '${item.name}'`);
      names.add(item.name);
      items.push(item);
    }
  }

  for (const item of items) {
    const dependencies = item.deps || [];
    if (!Array.isArray(dependencies)) throw new Error(`'${item.name}' dependencies must be an array`);
    for (const dependency of dependencies) {
      if (!names.has(dependency)) throw new Error(`'${item.name}' depends on unknown item '${dependency}'`);
    }
  }
  return plan;
}

function taskStart(task, tasks, resolved, resolving) {
  const spec = task.start;
  if (typeof spec === "string") return resolveTask(spec, tasks, resolved, resolving).end;
  if (!Array.isArray(spec)) throw new Error(`'${task.name}' needs a start date`);
  if (spec[0] === "date") return parseDate(spec[1]);
  if (spec[0] === "after") {
    const parent = resolveTask(spec[1], tasks, resolved, resolving);
    return addDuration(parent.end, spec[2]);
  }
  throw new Error(`Unknown start form for '${task.name}'`);
}

function resolveTask(name, tasks, resolved, resolving) {
  if (resolved.has(name)) return resolved.get(name);
  const task = tasks.get(name);
  if (!task) throw new Error(`Unknown start dependency '${name}'`);
  if (resolving.has(name)) throw new Error(`Scheduling cycle includes '${name}'`);
  resolving.add(name);
  const start = taskStart(task, tasks, resolved, resolving);
  const end = addDuration(start, task.end);
  if (end <= start) throw new Error(`'${name}' must end after it starts`);
  const span = { task, start, end };
  resolving.delete(name);
  resolved.set(name, span);
  return span;
}

export function resolvePlan(plan) {
  validatePlan(plan);
  const tasks = new Map();
  const milestones = new Map();
  for (const [workstreamIndex, workstream] of plan.workstreams.entries()) {
    for (const [taskIndex, task] of workstream.tasks.entries()) {
      tasks.set(task.name, { ...task, workstreamIndex, taskIndex });
    }
    for (const [milestoneIndex, milestone] of (workstream.milestones || []).entries()) {
      milestones.set(milestone.name, {
        ...milestone,
        dateValue: parseDate(milestone.date),
        workstreamIndex,
        milestoneIndex,
      });
    }
  }

  const resolvedTasks = new Map();
  for (const name of tasks.keys()) resolveTask(name, tasks, resolvedTasks, new Set());
  for (const capacity of plan.capacity || []) {
    parseDate(capacity.from);
    if (capacity.to) parseDate(capacity.to);
    for (const growth of capacity.grows || []) parseDate(growth.date);
  }
  for (const annotation of plan.annotations || []) parseDate(annotation.date);
  return { tasks: resolvedTasks, milestones };
}

export function dateBounds(plan, resolved) {
  const dates = [];
  for (const span of resolved.tasks.values()) dates.push(span.start, span.end);
  for (const milestone of resolved.milestones.values()) dates.push(milestone.dateValue);
  for (const capacity of plan.capacity || []) {
    dates.push(parseDate(capacity.from));
    if (capacity.to) dates.push(parseDate(capacity.to));
    for (const growth of capacity.grows || []) dates.push(parseDate(growth.date));
  }
  for (const annotation of plan.annotations || []) dates.push(parseDate(annotation.date));

  if (!dates.length) {
    const now = new Date();
    dates.push(new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1)));
    dates.push(new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 3, 1)));
  }
  const min = new Date(Math.min(...dates.map(Number)) - 7 * DAY);
  const max = new Date(Math.max(...dates.map(Number)) + 14 * DAY);
  min.setUTCDate(1);
  max.setUTCMonth(max.getUTCMonth() + 1, 1);
  return { min, max, days: Math.max(1, (max - min) / DAY) };
}

export function monthTicks(bounds) {
  const ticks = [];
  const cursor = new Date(Date.UTC(bounds.min.getUTCFullYear(), bounds.min.getUTCMonth(), 1));
  while (cursor <= bounds.max) {
    ticks.push(new Date(cursor));
    cursor.setUTCMonth(cursor.getUTCMonth() + 1);
  }
  return ticks;
}

export function capacityAt(capacity, date) {
  if (date < parseDate(capacity.from) || (capacity.to && date >= parseDate(capacity.to))) return 0;
  let chips = capacity.chips || 0;
  for (const growth of [...(capacity.grows || [])].sort((a, b) => a.date.localeCompare(b.date))) {
    if (parseDate(growth.date) <= date) chips = growth.to;
  }
  return chips;
}

export function capacitySegments(capacity, resolved, bounds) {
  const points = new Set([Number(bounds.min), Number(bounds.max), Number(parseDate(capacity.from))]);
  if (capacity.to) points.add(Number(parseDate(capacity.to)));
  for (const growth of capacity.grows || []) points.add(Number(parseDate(growth.date)));
  for (const span of resolved.tasks.values()) {
    if (span.task.cluster === capacity.name) {
      points.add(Number(span.start));
      points.add(Number(span.end));
    }
  }
  const sorted = [...points].filter((point) => point >= bounds.min && point <= bounds.max).sort((a, b) => a - b);
  const segments = [];
  for (let index = 0; index < sorted.length - 1; index += 1) {
    const start = new Date(sorted[index]);
    const end = new Date(sorted[index + 1]);
    if (end <= start) continue;
    const middle = new Date((Number(start) + Number(end)) / 2);
    const available = capacityAt(capacity, middle);
    let used = 0;
    for (const span of resolved.tasks.values()) {
      if (span.task.cluster === capacity.name && span.start < end && span.end > start) {
        used += typeof span.task.chips === "number" ? span.task.chips : available;
      }
    }
    segments.push({ start, end, available, used, over: used > available && used > 0 });
  }
  return segments;
}

export function decodeImport(value) {
  const input = value.trim();
  if (!input) throw new Error("Paste a Plantt link or chart JSON");
  if (input.startsWith("{")) return validatePlan(JSON.parse(input));

  let encoded = input;
  try {
    const url = new URL(input);
    encoded = url.hash.slice(1) || url.search.slice(1);
  } catch {
    encoded = input.replace(/^[?#]/, "");
  }
  const decoded = decompressFromEncodedURIComponent(encoded);
  if (!decoded) throw new Error("That link does not contain a Plantt chart");
  const payload = JSON.parse(decoded);
  return validatePlan(payload.d || payload);
}
