// Runs the applet scripts under a stub DOM and verifies the tree layout:
// every visible node placed, every visible parent->child pair drawn exactly
// once, and every arrow terminating at the child's final top-center.
const fs = require("fs");
const vm = require("vm");

const ROOT = __dirname + "/";
const graphJson = fs.readFileSync(ROOT + "graph-data.json", "utf8");

const selectors = {};
const byId = {};
function makeEl(sel) {
  const id = sel && sel.startsWith("#") ? sel.slice(1) : "";
  return {
    _id: id,
    style: { setProperty() {} },
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    dataset: {},
    value: id.includes("search") ? "" : "all",
    max: "100",
    hidden: false,
    offsetHeight: 100,
    offsetWidth: 100,
    clientWidth: 1200,
    clientHeight: 800,
    innerHTML: "",
    textContent: id === "graph-data" ? graphJson : "",
    append() {},
    addEventListener() {},
    setAttribute() {},
    scrollTo() {},
    getBoundingClientRect: () => ({ top: 100, left: 0, width: 1200, height: 800 }),
  };
}
const document = {
  getElementById: (id) => (byId[id] ||= makeEl("#" + id)),
  querySelector: (sel) => (selectors[sel] ||= makeEl(sel)),
  querySelectorAll: () => [],
  createElement: () => makeEl(""),
  addEventListener() {},
};
// A real browser fires rAF after all deferred scripts have executed; defer
// the callbacks until the three scripts finish loading.
const rafQueue = [];
const context = {
  document,
  innerWidth: 1200,
  innerHeight: 800,
  window: {
    addEventListener() {},
    matchMedia: () => ({ matches: false }),
    innerWidth: 1200,
    innerHeight: 800,
  },
  requestAnimationFrame: (fn) => rafQueue.push(fn) && rafQueue.length,
  __flushRaf: () => {
    let guard = 0;
    while (rafQueue.length && guard++ < 20) {
      const q = rafQueue.splice(0);
      for (const fn of q) fn();
    }
  },
  console,
  Math,
  Set,
  Map,
  JSON,
  Number,
  Date,
  Intl,
  Array,
  Object,
  String,
};
vm.createContext(context);
for (const f of ["app.js", "search.js", "timeline.js"]) {
  vm.runInContext(fs.readFileSync(ROOT + f, "utf8"), context, { filename: f });
}
// Drain load-time rAFs (browser fires these after all scripts execute).
context.__flushRaf();

const verify = `
(() => {
  const out = [];
  for (const fam of ["all", "2.7T", "5.7T", "10T"]) {
    document.querySelector("#family").value = fam;
    render();
    __flushRaf();
    const vis = treeMembers();
    const unplaced = vis.filter((n) => !positions.has(n.id)).map((n) => n.id);
    const html = document.querySelector("#paths").innerHTML;
    const arrows = [...html.matchAll(/data-from="([^"]*)" data-to="([^"]*)" d="([^"]*)"/g)];
    const visIds = new Set(vis.map((n) => n.id));
    const expected = [];
    for (const n of vis) for (const p of n.parents) if (visIds.has(p)) expected.push(p + "=>" + n.id);
    const got = arrows.map((m) => m[1] + "=>" + m[2]);
    const missing = expected.filter((k) => !got.includes(k));
    const dupes = got.filter((k, i) => got.indexOf(k) !== i);
    const badEnd = [];
    for (const m of arrows) {
      const child = positions.get(m[2]);
      if (!child) continue;
      const d = m[3].match(/H([\\d.]+) V([\\d.]+)$/);
      if (!d) { badEnd.push(m[2] + " unparsable"); continue; }
      const tx = Number(d[1]), ty = Number(d[2]);
      if (Math.abs(tx - (child.x + child.width / 2)) > 1e-6 || Math.abs(ty - child.y) > 1e-6)
        badEnd.push(m[1] + " => " + m[2] + " ends at " + tx + "," + ty + " but child is at " + (child.x + child.width / 2) + "," + child.y);
    }
    out.push({ fam, visible: vis.length, placed: positions.size, unplaced, arrows: arrows.length, expected: expected.length, missing, dupes, badEnd });
  }
  return JSON.stringify(out, null, 1);
})()
`;
const result = JSON.parse(vm.runInContext(verify, context));
let fail = false;
for (const r of result) {
  const bad =
    r.unplaced.length || r.missing.length || r.dupes.length || r.badEnd.length || r.arrows !== r.expected;
  if (bad) fail = true;
  console.log(
    `[${r.fam}] visible=${r.visible} placed=${r.placed} arrows=${r.arrows}/${r.expected}`,
    bad ? "FAIL" : "ok",
  );
  for (const k of ["unplaced", "missing", "dupes", "badEnd"])
    if (r[k].length) console.log("  " + k + ":", r[k].slice(0, 6).join("\n  "));
}
process.exit(fail ? 1 : 0);
