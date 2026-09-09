<script setup>
import { computed } from "vue";

import {
  PALETTE,
  MILLISECONDS_PER_DAY,
  capacitySegments,
  dateBounds,
  formatDate,
  monthTicks,
  parseDate,
  resolvePlan,
} from "../model.js";

const props = defineProps({
  plan: { type: Object, required: true },
  selected: { type: String, default: "" },
  showToday: { type: Boolean, default: true },
  showDependencies: { type: Boolean, default: true },
});

defineEmits(["select"]);

const LABEL_WIDTH = 190;
const RIGHT_PAD = 44;
const ROW_HEIGHT = 32;
const SECTION_GAP = 18;
const TOP = 56;

const timeline = computed(() => {
  const chartPlan = props.plan.title ? props.plan : { ...props.plan, title: "Untitled plan" };
  const resolved = resolvePlan(chartPlan);
  const bounds = dateBounds(props.plan, resolved);
  const width = Math.max(1080, LABEL_WIDTH + RIGHT_PAD + bounds.days * 3.6);
  const plotWidth = width - LABEL_WIDTH - RIGHT_PAD;
  const x = (date) => LABEL_WIDTH + ((date - bounds.min) / (bounds.days * MILLISECONDS_PER_DAY)) * plotWidth;

  let y = TOP;
  const sections = [];
  const tasks = [];
  const milestones = [];
  const positions = new Map();

  for (const [workstreamIndex, workstream] of props.plan.workstreams.entries()) {
    const top = y;
    y += 24;
    for (const [taskIndex, task] of workstream.tasks.entries()) {
      const span = resolved.tasks.get(task.name);
      const row = {
        ...task,
        kind: "task",
        workstreamIndex,
        taskIndex,
        y: y + ROW_HEIGHT / 2,
        x1: x(span.start),
        x2: x(span.end),
        startDate: formatDate(span.start),
        endDate: formatDate(span.end),
        color: PALETTE[workstreamIndex % PALETTE.length],
      };
      tasks.push(row);
      positions.set(task.name, row);
      y += ROW_HEIGHT;
    }
    for (const [milestoneIndex, milestone] of (workstream.milestones || []).entries()) {
      const date = resolved.milestones.get(milestone.name).dateValue;
      const row = {
        ...milestone,
        kind: "milestone",
        workstreamIndex,
        milestoneIndex,
        y: y + ROW_HEIGHT / 2,
        x: x(date),
        dateValue: date,
      };
      milestones.push(row);
      positions.set(milestone.name, row);
      y += ROW_HEIGHT;
    }
    if (!workstream.tasks.length && !(workstream.milestones || []).length) y += ROW_HEIGHT;
    sections.push({
      name: workstream.name,
      note: workstream.note,
      top,
      bottom: y,
      labelY: top + 15,
    });
    y += SECTION_GAP;
  }

  const capacityTop = y;
  const capacityLanes = [];
  if ((props.plan.capacity || []).length) {
    y += 34;
    for (const capacity of props.plan.capacity) {
      const center = y + 12;
      capacityLanes.push({
        ...capacity,
        y: center,
        x1: x(parseDate(capacity.from)),
        x2: x(capacity.to ? parseDate(capacity.to) : bounds.max),
        segments: capacitySegments(capacity, resolved, bounds).map((segment) => ({
          ...segment,
          x: x(segment.start),
          width: Math.max(0, x(segment.end) - x(segment.start)),
        })),
      });
      y += 34;
    }
  }
  const capacityBottom = y;

  const dependencies = [];
  for (const item of [...tasks, ...milestones]) {
    for (const dependencyName of item.deps || []) {
      const source = positions.get(dependencyName);
      if (!source) continue;
      const sourceX = source.kind === "task" ? source.x2 : source.x;
      const targetX = item.kind === "task" ? item.x1 : item.x;
      const bend = Math.max(18, Math.abs(targetX - sourceX) * 0.45);
      dependencies.push({
        key: `${dependencyName}:${item.name}`,
        source: dependencyName,
        target: item.name,
        violated: sourceX > targetX,
        path: `M ${sourceX} ${source.y} C ${sourceX + bend} ${source.y}, ${targetX - bend} ${item.y}, ${targetX} ${item.y}`,
      });
    }
  }

  const sectionByName = new Map(sections.map((section) => [section.name, section]));
  const annotations = (props.plan.annotations || []).flatMap((annotation, index) => {
    const date = parseDate(annotation.date);
    const target = positions.get(annotation.target);
    const section = sectionByName.get(annotation.target);
    let top = TOP;
    let bottom = capacityBottom;
    if (target) top = bottom = target.y;
    else if (section) ({ top, bottom } = section);
    else if (annotation.target === "@compute") {
      top = capacityTop;
      bottom = capacityBottom;
    } else return [];
    return [{ ...annotation, index, x: x(date), top, bottom }];
  });

  const today = new Date();
  const todayUtc = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));
  return {
    width,
    height: Math.max(270, y + 42),
    bounds,
    months: monthTicks(bounds).map((date) => ({ date, x: x(date) })),
    sections,
    tasks,
    milestones,
    dependencies,
    annotations,
    capacityLanes,
    capacityTop,
    capacityBottom,
    todayX: todayUtc >= bounds.min && todayUtc <= bounds.max ? x(todayUtc) : null,
  };
});

function monthLabel(date) {
  const month = date.toLocaleDateString(undefined, { month: "short", timeZone: "UTC" });
  return date.getUTCMonth() === 0 ? `${month} ${date.getUTCFullYear()}` : month;
}

function tooltip(item) {
  if (item.kind === "task") {
    const compute = item.cluster ? ` · ${item.cluster}${item.chips ? ` (${item.chips} chips)` : ""}` : "";
    return `${item.name}\n${item.startDate} → ${item.endDate}${compute}${item.tooltip ? `\n${item.tooltip}` : ""}`;
  }
  return `${item.name}\n${item.date}${item.tooltip ? `\n${item.tooltip}` : ""}`;
}
</script>

<template>
  <div class="gantt-scroll">
    <svg
      class="gantt"
      :viewBox="`0 0 ${timeline.width} ${timeline.height}`"
      :style="{ minWidth: `${timeline.width}px` }"
      role="img"
      :aria-label="`${plan.title} project timeline`"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" class="arrow-head" />
        </marker>
      </defs>

      <g class="months">
        <g v-for="month in timeline.months" :key="month.date.toISOString()">
          <line :x1="month.x" :x2="month.x" y1="36" :y2="timeline.height - 22" />
          <text :x="month.x + 7" y="25">{{ monthLabel(month.date) }}</text>
        </g>
      </g>

      <line
        v-if="showToday && timeline.todayX"
        class="today-line"
        :x1="timeline.todayX"
        :x2="timeline.todayX"
        y1="36"
        :y2="timeline.height - 22"
      />

      <g v-if="showDependencies" class="dependencies">
        <path
          v-for="dependency in timeline.dependencies"
          :key="dependency.key"
          :d="dependency.path"
          :class="{ violated: dependency.violated }"
          marker-end="url(#arrow)"
        >
          <title>{{ dependency.source }} → {{ dependency.target }}{{ dependency.violated ? " (schedule violation)" : "" }}</title>
        </path>
      </g>

      <g v-for="section in timeline.sections" :key="section.name" class="workstream">
        <text x="18" :y="section.labelY" class="workstream-name">{{ section.name }}</text>
        <text v-if="section.note" x="18" :y="section.labelY + 17" class="workstream-note">{{ section.note }}</text>
      </g>

      <g
        v-for="task in timeline.tasks"
        :key="task.name"
        class="task-mark"
        :class="{ selected: selected === task.name }"
        tabindex="0"
        role="button"
        @click="$emit('select', task)"
        @keydown.enter="$emit('select', task)"
      >
        <rect
          :x="task.x1"
          :y="task.y - Math.max(5, Math.min(10, task.significance || 6))"
          :width="Math.max(3, task.x2 - task.x1)"
          :height="Math.max(10, Math.min(20, (task.significance || 6) * 2))"
          rx="4"
          :fill="task.color"
        />
        <text :x="task.x1 - 8" :y="task.y + 4" text-anchor="end" class="item-label">{{ task.name }}</text>
        <text v-if="task.cluster" :x="task.x2 + 8" :y="task.y + 4" class="cluster-label">{{ task.cluster }}</text>
        <title>{{ tooltip(task) }}</title>
      </g>

      <g
        v-for="milestone in timeline.milestones"
        :key="milestone.name"
        class="milestone-mark"
        :class="{ selected: selected === milestone.name }"
        tabindex="0"
        role="button"
        @click="$emit('select', milestone)"
        @keydown.enter="$emit('select', milestone)"
      >
        <line
          v-if="milestone.line"
          :x1="milestone.x"
          :x2="milestone.x"
          y1="36"
          :y2="timeline.height - 22"
          :stroke="milestone.line"
          class="milestone-line"
        />
        <text v-if="milestone.emoji" :x="milestone.x" :y="milestone.y + 6" text-anchor="middle" class="milestone-emoji">{{ milestone.emoji }}</text>
        <rect v-else :x="milestone.x - 6" :y="milestone.y - 6" width="12" height="12" class="milestone-diamond" transform-origin="center" />
        <text :x="milestone.x + 12" :y="milestone.y + 5" class="item-label milestone-label">{{ milestone.name }}</text>
        <title>{{ tooltip(milestone) }}</title>
      </g>

      <g v-if="timeline.capacityLanes.length" class="capacity">
        <text x="18" :y="timeline.capacityTop + 18" class="workstream-name">Compute capacity</text>
        <text x="18" :y="timeline.capacityTop + 35" class="workstream-note">filled = in use · red = over-subscribed</text>
        <g v-for="lane in timeline.capacityLanes" :key="lane.name">
          <text x="18" :y="lane.y + 4" class="capacity-label">{{ lane.name }}</text>
          <line :x1="lane.x1" :x2="lane.x2" :y1="lane.y" :y2="lane.y" class="capacity-base" />
          <rect
            v-for="segment in lane.segments"
            :key="`${segment.start.toISOString()}:${segment.end.toISOString()}`"
            :x="segment.x"
            :y="lane.y - 5"
            :width="segment.width"
            height="10"
            :fill="segment.over ? 'var(--warn)' : lane.color || 'var(--mark)'"
            :opacity="segment.used ? Math.max(0.32, Math.min(1, segment.used / Math.max(1, segment.available))) : 0"
          >
            <title>{{ lane.name }}: {{ segment.used }} / {{ segment.available }} chips</title>
          </rect>
        </g>
      </g>

      <g v-for="annotation in timeline.annotations" :key="annotation.index" class="annotation">
        <line
          :x1="annotation.x"
          :x2="annotation.x"
          :y1="annotation.top"
          :y2="annotation.bottom"
          :stroke="annotation.color || 'var(--muted)'"
        />
        <text
          :x="annotation.x + 5"
          :y="annotation.edge === 'top' ? annotation.top - 5 : annotation.bottom + 15"
          :fill="annotation.color || 'var(--muted)'"
        >{{ annotation.text }}</text>
      </g>
    </svg>
  </div>
</template>
