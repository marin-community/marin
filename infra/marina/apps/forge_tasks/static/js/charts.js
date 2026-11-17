// Small SVG charts: labelled horizontal bars, stacked bars, column histograms, heatmaps and the pipeline diagram.

import { h, s, tip, tipBody, compact, hideTip, percent } from "./util.js";

const LABEL_W = 130;
const ROW_H = 26;
const BAR_H = 14;

function niceMax(value) {
  if (value <= 0) return 1;
  const exp = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) if (value <= step * exp) return step * exp;
  return 10 * exp;
}

function barPath(x, y, w, hgt, r = 4) {
  if (w <= 0) return "";
  const rr = Math.min(r, w, hgt / 2);
  return `M${x},${y}h${w - rr}a${rr},${rr} 0 0 1 ${rr},${rr}v${hgt - 2 * rr}a${rr},${rr} 0 0 1 -${rr},${rr}h-${w - rr}z`;
}

// rows: [{label, value, color, tip, onClick, note}]
export function hbars(rows, { format = compact, width = 520, labelWidth = LABEL_W, empty = "Nothing to show." } = {}) {
  if (!rows.length || rows.every((r) => !r.value)) return h("div", { class: "empty inline" }, empty);
  const max = Math.max(...rows.map((r) => r.value));
  const valueW = 70;
  const plotW = width - labelWidth - valueW;
  const height = rows.length * ROW_H + 4;
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  rows.forEach((row, i) => {
    const y = i * ROW_H + 4;
    const w = max ? (row.value / max) * plotW : 0;
    const g = s("g", { style: row.onClick ? "cursor:pointer" : "" });
    g.append(
      s("text", { x: labelWidth - 10, y: y + BAR_H - 2, "text-anchor": "end", class: "ink2" }, row.label),
      s("path", { d: barPath(labelWidth, y, Math.max(w, row.value ? 3 : 0), BAR_H), fill: row.color || "var(--accent)" }),
      s("text", { x: labelWidth + w + 6, y: y + BAR_H - 2, class: "ink" }, format(row.value)),
      s("rect", { class: "hit", x: 0, y: y - 5, width, height: ROW_H }),
    );
    if (row.onClick) g.addEventListener("click", () => { hideTip(); row.onClick(); });
    tip(g, row.tip || tipBody(row.label, format(row.value)));
    svg.append(g);
  });
  return h("div", { class: "chart" }, svg);
}

// rows: [{label, parts: [{key, value, color, tip}], onClick}]
export function stackedHbars(rows, { width = 520, format = compact, total = null } = {}) {
  if (!rows.length) return h("div", { class: "empty inline" }, "Nothing to show.");
  const max = total ?? Math.max(...rows.map((r) => r.parts.reduce((a, p) => a + p.value, 0)));
  const valueW = 60;
  const plotW = width - LABEL_W - valueW;
  const height = rows.length * ROW_H + 4;
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  rows.forEach((row, i) => {
    const y = i * ROW_H + 4;
    svg.append(s("text", { x: LABEL_W - 10, y: y + BAR_H - 2, "text-anchor": "end", class: "ink2" }, row.label));
    let x = LABEL_W;
    const sum = row.parts.reduce((a, p) => a + p.value, 0);
    const visible = row.parts.filter((p) => p.value > 0);
    visible.forEach((part, j) => {
      const w = max ? (part.value / max) * plotW : 0;
      const gap = j < visible.length - 1 ? 2 : 0;
      const first = j === 0;
      const lastPart = j === visible.length - 1;
      const r = 4;
      const ww = Math.max(w - gap, 1.5);
      let d;
      if (first && lastPart) d = barPath(x, y, ww, BAR_H, r);
      else if (lastPart) d = barPath(x, y, ww, BAR_H, r);
      else d = `M${x},${y}h${ww}v${BAR_H}h-${ww}z`;
      const rr = Math.min(r, ww / 2);
      if (first && !lastPart) d = `M${x + rr},${y}h${ww - rr}v${BAR_H}h-${ww - rr}a${rr},${rr} 0 0 1 -${rr},-${rr}v-${BAR_H - 2 * rr}a${rr},${rr} 0 0 1 ${rr},-${rr}z`;
      const seg = s("path", { d, fill: part.color });
      tip(seg, part.tip || tipBody(part.key, format(part.value)));
      if (row.onClick || part.onClick) {
        seg.style.cursor = "pointer";
        seg.addEventListener("click", () => { hideTip(); (part.onClick || row.onClick)(); });
      }
      svg.append(seg);
      x += w;
    });
    svg.append(s("text", { x: x + 6, y: y + BAR_H - 2, class: "ink" }, format(sum)));
  });
  return h("div", { class: "chart" }, svg);
}


// bins: [{label, value, color, tip, onClick, active}]; vertical columns with labels under them.
export function columns(bins, { width = 560, height = 210, format = compact, xLabel = "", shade = null } = {}) {
  if (!bins.length || bins.every((b) => !b.value)) return h("div", { class: "empty inline" }, "Nothing to show.");
  const top = 18;
  const bottom = xLabel ? 42 : 26;
  const left = 8;
  const plotH = height - top - bottom;
  const slot = (width - left * 2) / bins.length;
  const barW = Math.min(56, slot * 0.72);
  const max = Math.max(...bins.map((b) => b.value));
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  if (shade) {
    // shade: [firstIndex, lastIndex] of bins drawn over a band background.
    const x0 = left + shade[0] * slot;
    const x1 = left + (shade[1] + 1) * slot;
    svg.append(s("rect", { x: x0, y: top - 6, width: x1 - x0, height: plotH + 6, fill: "var(--band)", rx: 6 }));
    if (shade[2]) svg.append(s("text", { x: (x0 + x1) / 2, y: top + 6, "text-anchor": "middle", class: "ink2 small-svg" }, shade[2]));
  }
  svg.append(s("line", { x1: left, x2: width - left, y1: top + plotH, y2: top + plotH, stroke: "var(--axis)" }));
  bins.forEach((bin, i) => {
    const hgt = max ? (bin.value / max) * (plotH - 14) : 0;
    const x = left + i * slot + (slot - barW) / 2;
    const y = top + plotH - hgt;
    const g = s("g", { class: `col${bin.active ? " active" : ""}`, style: bin.onClick ? "cursor:pointer" : "" });
    g.append(
      s("rect", { class: "hit", x: left + i * slot, y: top - 6, width: slot, height: plotH + bottom }),
      s("rect", { x, y, width: barW, height: Math.max(hgt, bin.value ? 2 : 0), rx: 3, fill: bin.color || "var(--accent)", opacity: bin.dim ? 0.35 : 1 }),
      s("text", { x: x + barW / 2, y: y - 5, "text-anchor": "middle", class: "ink small-svg" }, bin.value ? format(bin.value) : ""),
      s("text", { x: x + barW / 2, y: top + plotH + 16, "text-anchor": "middle", class: "ink2 small-svg" }, bin.label),
    );
    if (bin.onClick) g.addEventListener("click", () => { hideTip(); bin.onClick(); });
    tip(g, bin.tip || tipBody(bin.label, format(bin.value)));
    svg.append(g);
  });
  if (xLabel) svg.append(s("text", { x: width / 2, y: height - 6, "text-anchor": "middle", class: "muted small-svg" }, xLabel));
  return h("div", { class: "chart" }, svg);
}

// rows/cols: [{key, label}]; value(rowKey, colKey) -> number. Cells shaded on a single-hue ramp.
export function heatmap(rows, cols, value, { onClick, tipFor, colWidth = 92, labelWidth = 150, hue = "var(--critical)" } = {}) {
  const cellH = 30;
  const head = 54;
  const width = labelWidth + cols.length * colWidth;
  const height = head + rows.length * cellH;
  const values = rows.flatMap((r) => cols.map((c) => value(r.key, c.key)));
  const max = Math.max(1, ...values);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", class: "heatmap" });
  cols.forEach((c, j) => {
    const x = labelWidth + j * colWidth + colWidth / 2;
    const words = c.label.split(" ");
    const mid = Math.ceil(words.length / 2);
    svg.append(s("text", { x, y: head - 26, "text-anchor": "middle", class: "ink2 small-svg" }, words.slice(0, mid).join(" ")));
    svg.append(s("text", { x, y: head - 12, "text-anchor": "middle", class: "ink2 small-svg" }, words.slice(mid).join(" ")));
  });
  rows.forEach((r, i) => {
    const y = head + i * cellH;
    svg.append(s("text", { x: labelWidth - 10, y: y + cellH / 2 + 4, "text-anchor": "end", class: "ink2" }, r.label));
    cols.forEach((c, j) => {
      const v = value(r.key, c.key);
      const x = labelWidth + j * colWidth;
      const g = s("g", { style: onClick && v ? "cursor:pointer" : "" });
      g.append(
        s("rect", { x: x + 2, y: y + 2, width: colWidth - 4, height: cellH - 4, rx: 4, fill: "var(--surface-2)" }),
        s("rect", { x: x + 2, y: y + 2, width: colWidth - 4, height: cellH - 4, rx: 4, fill: hue, opacity: v ? 0.12 + 0.88 * Math.sqrt(v / max) : 0 }),
        s("text", { x: x + colWidth / 2, y: y + cellH / 2 + 4, "text-anchor": "middle", class: v / max > 0.45 ? "on-dark small-svg" : "ink small-svg" }, v ? compact(v) : "·"),
      );
      if (onClick && v) g.addEventListener("click", () => { hideTip(); onClick(r.key, c.key); });
      tip(g, tipFor ? tipFor(r, c, v) : tipBody(`${r.label} · ${c.label}`, compact(v)));
      svg.append(g);
    });
  });
  return h("div", { class: "chart" }, svg);
}

// Eight dots for the eight confirmation trials: filled when the trial passed.
export function trialDots(passes, total = 8, { size = 9, gap = 3 } = {}) {
  const width = total * (size + gap) - gap;
  const svg = s("svg", { viewBox: `0 0 ${width} ${size}`, width, height: size, class: "trialdots" });
  for (let i = 0; i < total; i++) {
    const pass = i < passes;
    svg.append(s("circle", { cx: i * (size + gap) + size / 2, cy: size / 2, r: size / 2 - 0.75, fill: pass ? "var(--good)" : "none", stroke: pass ? "var(--good)" : "var(--border-strong)", "stroke-width": 1.5 }));
  }
  return svg;
}

// The forge pipeline: how one task is built and accepted, with counts from the selected tasks.
// stats: {capabilities, tasks, revised, rounds, fa, fr, ctl, review, band, confirmed}; onPick(key) filters the task list.
export function pipeline(stats, { onPick, active } = {}) {
  const W = 1200;
  const H = 360;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const defs = s("defs", {}, s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--muted)" })));
  svg.append(defs);
  const node = (key, x, y, w, hgt, label, count, sub, color) => {
    const g = s("g", { class: `node${active === key ? " active" : ""}`, transform: `translate(${x},${y})` });
    g.append(
      s("rect", { width: w, height: hgt, rx: 10 }),
      s("rect", { class: "stripe", width: 4, height: hgt - 16, x: 0, y: 8, rx: 2, fill: color }),
      s("text", { class: "label", x: 14, y: 22 }, label),
      count !== null ? s("text", { class: "count", x: 14, y: 40 }, count) : null,
      sub ? s("text", { class: "sub", x: 14, y: hgt - 12 }, sub) : null,
    );
    if (onPick && key) g.addEventListener("click", () => { hideTip(); onPick(key); });
    else g.style.cursor = "default";
    return g;
  };
  const edge = (x1, y1, x2, y2) => s("path", { class: "edge live", d: `M${x1},${y1} L${x2},${y2}`, "stroke-width": 2.5, "marker-end": "url(#arrow)" });
  const cy = 150;
  svg.append(
    edge(150, cy, 182, cy), edge(332, cy, 364, cy), edge(514, cy, 546, cy), edge(846, cy, 878, cy), edge(949, cy + 40, 949, 252),
  );
  svg.append(
    node(null, 10, cy - 40, 140, 80, "Capability", `${compact(stats.capabilities)} covered`, "of 1,999 in the catalog", "var(--phase-triage)"),
    node(null, 184, cy - 40, 148, 80, "Proposal", "a plan for one task", "what to build, how to grade", "var(--phase-triage)"),
    node(null, 366, cy - 40, 148, 80, "Author", "an agent builds it", "files, grader, controls", "var(--phase-author)"),
  );
  // The validation round: five checks run against every submission.
  const vx = 548;
  const vy = 36;
  const vw = 298;
  const box = s("g", {});
  box.append(s("rect", { x: vx, y: vy, width: vw, height: 236, rx: 12, fill: "var(--surface-2)", stroke: "var(--border-strong)" }));
  box.append(s("text", { x: vx + 14, y: vy + 22, class: "ink", style: "font-weight:600;font-size:13px" }, "Validation round"));
  box.append(s("text", { x: vx + 14, y: vy + 38, class: "muted small-svg" }, "every submission, every revision"));
  svg.append(box);
  const checks = [
    ["ctl", "Controls replay", "known answers graded as expected", `${compact(stats.ctl)} caught`, "var(--phase-controls)"],
    [null, "3 solver trials", "GLM-5.3 attempts the task", null, "var(--phase-solver)"],
    ["fa", "False-accept hunter", "a wrong answer the grader credits", `${compact(stats.fa)} caught`, "var(--critical)"],
    ["fr", "False-reject hunter", "a right answer the grader refuses", `${compact(stats.fr)} caught`, "var(--serious)"],
    ["review", "Transcript review", "reads trials for task defects", `${compact(stats.review)} caught`, "var(--phase-adversary)"],
  ];
  checks.forEach(([key, label, sub, count, color], i) => {
    const y = vy + 50 + i * 36;
    const g = s("g", { class: `check${active === key ? " active" : ""}`, transform: `translate(${vx + 10},${y})`, style: key && onPick ? "cursor:pointer" : "" });
    g.append(
      s("rect", { width: vw - 20, height: 30, rx: 7, class: "check-bg" }),
      s("circle", { cx: 14, cy: 15, r: 5, fill: color }),
      s("text", { x: 26, y: 13, class: "ink", style: "font-size:12px;font-weight:600" }, label),
      s("text", { x: 26, y: 25, class: "muted", style: "font-size:10.5px" }, sub),
      count ? s("text", { x: vw - 30, y: 13, "text-anchor": "end", class: "ink2", style: "font-size:11.5px" }, count) : null,
    );
    if (key && onPick) g.addEventListener("click", () => { hideTip(); onPick(key); });
    if (count) tip(g, tipBody(label, `${count.replace(" caught", "")} accepted tasks had this check fail on an earlier version, which was then fixed.`));
    svg.append(g);
  });
  svg.append(node(null, 880, cy - 40, 138, 80, "Review", "a fixed rule decides", "accept, or revise", "var(--phase-build)"));
  // Revision loop: review back to author.
  const loop = s("path", { class: "loop live", d: `M949,${cy - 40} C949,6 440,6 440,${cy - 40}`, "marker-end": "url(#arrow)" });
  svg.append(loop);
  const loopLabel = s("g", { style: onPick ? "cursor:pointer" : "" }, s("rect", { x: 600, y: 2, width: 206, height: 26, rx: 13, fill: "var(--surface)", stroke: "var(--serious)" }), s("text", { x: 703, y: 19, "text-anchor": "middle", class: "loop-label live" }, `revise: ${compact(stats.revised)} tasks, ${compact(stats.rounds)} rounds`));
  if (onPick) loopLabel.addEventListener("click", () => { hideTip(); onPick("revised"); });
  tip(loopLabel, tipBody("Revision rounds", `${compact(stats.revised)} of ${compact(stats.tasks)} tasks went back to their author at least once (${percent(stats.revised / stats.tasks)}).`, "Every finding goes back with the evidence; the author fixes it and the task is validated again.", `${compact(stats.band)} rounds only asked to make the task harder or easier.`));
  svg.append(loopLabel);
  svg.append(node(null, 880, 254, 138, 80, "Confirmation", "5 more trials", "and a final review", "var(--phase-solver)"));
  svg.append(edge(1018, 294, 1048, 294));
  svg.append(node(null, 1050, 254, 140, 80, "Accepted", `${compact(stats.tasks)} tasks`, "8 graded trials each", "var(--good)"));
  return h("div", { class: "pipeline" }, svg);
}
