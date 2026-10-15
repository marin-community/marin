<script setup>
import { computed, onBeforeUnmount, ref, toRaw, watch } from "vue";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { examples } from "./examples";
import ImageGallery from "./ImageGallery.vue";

const OUTPUT_BASE = "https://output.invalid/";

const props = defineProps({ record: { type: Object, required: true } });
const emit = defineEmits(["back"]);
const project = ref(structuredClone(toRaw(props.record.document)));
const error = ref("");
const saved = ref("Saved to Postgres");
const revision = ref(props.record.revision);
const dirty = ref(false);
const resultRevision = ref(props.record.result_revision);
let lastResult = props.record.result;
let generation = 0;
let savedGeneration = 0;
let saveTimer;
let savingPromise;
const selected = ref(project.value.files[0]?.path);
const current = computed(() =>
  project.value.files.find((f) => f.path === selected.value),
);
const pythonFiles = computed(() =>
  project.value.files.filter((f) => f.path.endsWith(".py")),
);
const running = ref(false);
const result = ref(null);
const tab = ref("report");
const newName = ref("");
const pendingExample = ref("bayesian");
const showReplace = ref(false);
const uploadInput = ref(null);
const elapsed = ref(0);
let timer;
let urls = [];
const artifacts = ref([]);
watch(
  project,
  () => {
    generation++;
    dirty.value = true;
    saved.value = "Unsaved changes";
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveProject, 600);
  },
  { deep: true, flush: "sync" },
);
/** Return true once the current edits are saved (or unchanged), and false after a save failure. */
async function saveProject() {
  clearTimeout(saveTimer);
  if (savingPromise) return savingPromise;
  savingPromise = (async () => {
    try {
      while (savedGeneration !== generation) {
        saved.value = "Saving…";
        const snapshotGeneration = generation;
        const response = await fetch(`api/projects/${props.record.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            revision: revision.value,
            document: project.value,
          }),
        });
        if (!response.ok) {
          const body = await response.json();
          throw new Error(body.detail || "Save failed");
        }
        const record = await response.json();
        revision.value = record.revision;
        savedGeneration = snapshotGeneration;
      }
      dirty.value = false;
      saved.value = "Saved to Postgres";
      return true;
    } catch (e) {
      saved.value = "Save failed — your edits are still open";
      error.value = typeof e.message === "string" ? e.message : "Save failed";
      return false;
    }
  })();
  try {
    return await savingPromise;
  } finally {
    savingPromise = null;
  }
}
function beforeUnload(event) {
  if (dirty.value || running.value) {
    event.preventDefault();
    event.returnValue = "";
  }
}
window.addEventListener("beforeunload", beforeUnload);
defineExpose({ saveProject, running });
function validPath(path) {
  return (
    path &&
    !path.startsWith("/") &&
    !path.includes("\\") &&
    path.split("/").every((p) => p && p !== "." && p !== "..") &&
    path.split("/")[0] !== "output"
  );
}
function addFile() {
  const path = newName.value.trim();
  if (!validPath(path) || project.value.files.some((f) => f.path === path)) {
    error.value = "Choose a unique relative path outside output/.";
    return;
  }
  project.value.files.push({ path, content: "", encoding: "utf-8" });
  selected.value = path;
  newName.value = "";
  error.value = "";
}
function deleteFile() {
  project.value.files = project.value.files.filter(
    (f) => f.path !== selected.value,
  );
  selected.value = project.value.files[0]?.path;
}
async function upload(event) {
  try {
    for (const file of event.target.files) {
      const path = file.webkitRelativePath || file.name;
      if (!validPath(path)) throw new Error(`Invalid path: ${path}`);
      if (project.value.files.some((f) => f.path === path))
        throw new Error(
          `${path} already exists; delete it before uploading a replacement`,
        );
      const bytes = new Uint8Array(await file.arrayBuffer());
      let content, encoding;
      try {
        content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
        if (content.includes("\0")) throw new Error("Binary");
        encoding = "utf-8";
      } catch {
        let binary = "";
        for (const byte of bytes) binary += String.fromCharCode(byte);
        content = btoa(binary);
        encoding = "base64";
      }
      project.value.files.push({ path, content, encoding });
      selected.value = path;
    }
  } catch (e) {
    error.value = e.message;
  }
  event.target.value = "";
}
function clearResults() {
  urls.forEach((url) => URL.revokeObjectURL(url));
  urls = [];
  artifacts.value = [];
  result.value = null;
}
function loadExample() {
  project.value = structuredClone(examples[pendingExample.value]);
  selected.value = project.value.files[0].path;
  clearResults();
  error.value = "";
  showReplace.value = false;
}
function download(content, name, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportProject() {
  download(
    JSON.stringify(project.value, null, 2),
    "python-project.json",
    "application/json",
  );
}
function showResult(value) {
  clearResults();
  result.value = value;
  if (!value) return;
  artifacts.value = value.artifacts.map((artifact) => {
    const bytes = Uint8Array.from(atob(artifact.content), (c) =>
      c.charCodeAt(0),
    );
    const url = URL.createObjectURL(
      new Blob([bytes], { type: artifact.media_type }),
    );
    urls.push(url);
    return {
      ...artifact,
      url,
      previewUrl: `data:${artifact.media_type};base64,${artifact.content}`,
      text: new TextDecoder().decode(bytes),
    };
  });
}
showResult(lastResult);
async function run() {
  error.value = "";
  running.value = true;
  elapsed.value = 0;
  clearResults();
  tab.value = "report";
  timer = setInterval(() => elapsed.value++, 1000);
  try {
    if (!(await saveProject())) return;
    const response = await fetch(`api/projects/${props.record.id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: revision.value }),
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.detail || "Run failed");
    }
    const record = await response.json();
    lastResult = record.result;
    resultRevision.value = record.result_revision;
    showResult(lastResult);
    if (!artifacts.value.length) tab.value = "logs";
  } catch (e) {
    error.value = e.message;
    showResult(lastResult);
  } finally {
    if (!result.value) showResult(lastResult);
    running.value = false;
    clearInterval(timer);
  }
}
const markdownFile = computed(() =>
  artifacts.value.find((a) => a.path === "index.md"),
);
const images = computed(() =>
  artifacts.value.filter((a) => a.media_type.startsWith("image/")),
);
const report = computed(() => {
  if (!markdownFile.value) return "";
  const doc = new DOMParser().parseFromString(
    DOMPurify.sanitize(marked.parse(markdownFile.value.text)),
    "text/html",
  );
  for (const node of doc.querySelectorAll("img, a")) {
    const attr = node.tagName === "IMG" ? "src" : "href";
    const path = node.getAttribute(attr);
    if (!path) continue;
    // Resolve paths against output/index.md, including optional output/ prefixes.
    const normalized = new URL(path, OUTPUT_BASE).pathname.replace(
      /^\/output\//,
      "/",
    );
    const artifact = artifacts.value.find(
      (a) => new URL(a.path, OUTPUT_BASE).pathname === normalized,
    );
    if (artifact && !/^[a-z]+:/i.test(path)) {
      node.setAttribute(
        attr,
        attr === "src" ? artifact.previewUrl : artifact.url,
      );
      if (attr === "href")
        node.setAttribute("download", artifact.path.split("/").pop());
    } else if (attr === "href") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener");
    }
  }
  return doc.body.innerHTML;
});
onBeforeUnmount(() => {
  clearInterval(timer);
  clearTimeout(saveTimer);
  window.removeEventListener("beforeunload", beforeUnload);
  clearResults();
});
</script>

<template>
  <div class="app-shell">
    <header>
      <a
        class="brand"
        href="#"
        aria-label="Python playground projects"
        @click.prevent="emit('back')"
      >
        <span class="brand-icon">py</span>
        <div>
          <h1>Python playground</h1>
          <p>MARINA / EXPERIMENTS</p>
        </div>
      </a>
      <div class="header-actions">
        <span class="trusted">Trusted workspace · CPython</span>
        <button @click="emit('back')" :disabled="running">← Projects</button>
        <button @click="saveProject" :disabled="running || !dirty">Save</button>
        <button @click="exportProject">Export project ↓</button>
      </div>
    </header>
    <div class="project-bar">
      <input
        class="project-title"
        aria-label="Project title"
        v-model="project.title"
        :disabled="running"
      />
      <span class="save-status">{{ saved }}</span>
      <div class="example-control">
        <label for="example">Start from</label
        ><select id="example" v-model="pendingExample" :disabled="running">
          <option value="bayesian">Bayesian eval</option>
          <option value="seaborn">Seaborn report</option>
          <option value="storage">Object storage preview</option>
          <option value="blank">Blank project</option></select
        ><button @click="showReplace = true" :disabled="running">
          Load example
        </button>
      </div>
    </div>
    <div v-if="showReplace" class="notice">
      Replace the current files with {{ examples[pendingExample].title }}?
      Export first to keep a copy.
      <button @click="loadExample">Replace files</button
      ><button @click="showReplace = false">Cancel</button>
    </div>
    <div v-if="error" class="notice error" role="alert">
      {{ error }} <button @click="error = ''">Dismiss</button>
    </div>
    <main>
      <section class="workspace">
        <div class="section-heading">
          <span>01 / WORKSPACE</span
          ><span>{{ project.files.length }} files</span>
        </div>
        <div class="editor-layout">
          <aside class="file-sidebar">
            <div class="sidebar-title">
              FILES
              <button
                @click="uploadInput.click()"
                :disabled="running"
                title="Upload files"
              >
                Upload ↑
              </button>
            </div>
            <button
              v-for="file in project.files"
              :key="file.path"
              class="file-item"
              :class="{ active: selected === file.path }"
              @click="selected = file.path"
            >
              <span class="file-icon">{{
                file.path.endsWith(".py") ? "py" : "↳"
              }}</span
              >{{ file.path }}
            </button>
            <form @submit.prevent="addFile" class="add-file">
              <input
                aria-label="New file path"
                placeholder="helpers.py"
                v-model="newName"
                :disabled="running"
              /><button :disabled="running">+ Add file</button>
            </form>
            <div class="output-hint">
              <span>↳ output/</span>
              <p>Generated at run time.<br />Write reports and plots here.</p>
            </div>
          </aside>
          <div class="editor">
            <div class="editor-toolbar">
              <span>{{ current?.path || "Select a file" }}</span
              ><button v-if="current" @click="deleteFile" :disabled="running">
                Delete file
              </button>
            </div>
            <textarea
              v-if="current?.encoding === 'utf-8'"
              v-model="current.content"
              :aria-label="`Edit ${current.path}`"
              :disabled="running"
              spellcheck="false"
              class="code-editor"
              @keydown.tab.prevent="
                (e) => {
                  const start = e.target.selectionStart;
                  const end = e.target.selectionEnd;
                  current.content =
                    current.content.slice(0, start) +
                    '    ' +
                    current.content.slice(end);
                  $nextTick(() => {
                    e.target.selectionStart = e.target.selectionEnd = start + 4;
                  });
                }
              "
            ></textarea>
            <div v-else class="binary-notice">
              {{
                current
                  ? "Binary input uploaded. Your Python program can read this file by name."
                  : "Add or upload a file to start."
              }}
            </div>
            <div class="editor-footer">
              <span>{{
                current?.encoding === "utf-8"
                  ? "UTF-8 · Tab inserts 4 spaces"
                  : "Binary file"
              }}</span
              ><span>Full Python</span>
            </div>
          </div>
        </div>
        <div class="run-settings">
          <div>
            <label for="entrypoint">ENTRYPOINT</label
            ><select
              id="entrypoint"
              v-model="project.entrypoint"
              :disabled="running"
            >
              <option
                v-for="file in pythonFiles"
                :key="file.path"
                :value="file.path"
              >
                {{ file.path }}
              </option>
            </select>
            <p>Runs in the project directory.</p>
          </div>
          <div class="packages">
            <label for="packages"
              >EXTRA PACKAGES <span>one requirement per line</span></label
            ><textarea
              id="packages"
              v-model="project.packages"
              :disabled="running"
              placeholder="numpy&#10;seaborn>=0.13"
              rows="3"
              spellcheck="false"
            ></textarea>
          </div>
          <button
            class="run-button"
            @click="run"
            :disabled="
              running || !pythonFiles.some((f) => f.path === project.entrypoint)
            "
          >
            {{ running ? `Running · ${elapsed}s` : "▶ Run Python" }}
          </button>
        </div>
        <p class="execution-note">
          Includes marin-rigging, PyArrow and FineStore from Marina. Extra
          packages install with uv. Runs stop after 120s. Storage credentials
          are inherited from the host.
        </p>
      </section>
      <section class="results">
        <div class="section-heading">
          <span>02 / OUTPUT</span
          ><span
            v-if="result"
            :class="result.exit_code === 0 ? 'success' : 'failed'"
            >{{
              result.timed_out
                ? "Timed out"
                : result.exit_code === 0
                  ? "Completed"
                  : `Exit ${result.exit_code}`
            }}
            · {{ result.duration }}s</span
          ><span v-else>{{ running ? "Executing…" : "Ready to run" }}</span>
        </div>
        <div class="tabs">
          <button
            v-for="item in ['report', 'images', 'files', 'logs']"
            :key="item"
            @click="tab = item"
            :class="{ active: tab === item }"
          >
            {{ item
            }}<span v-if="item === 'files' && result">{{
              artifacts.length
            }}</span>
          </button>
        </div>
        <div
          v-if="result && (resultRevision !== revision || dirty)"
          class="warnings"
        >
          <p>
            Showing results from saved source revision {{ resultRevision }}. Run
            again to update them.
          </p>
        </div>
        <div v-if="result?.warnings.length" class="warnings">
          <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
        </div>
        <div class="preview-content">
          <template v-if="tab === 'report'">
            <article
              v-if="report"
              class="markdown-report"
              v-html="report"
            ></article>
            <div v-else class="empty-state">
              <div class="empty-icon">↗</div>
              <h2>
                {{
                  running
                    ? "Your program is running"
                    : "A place for your results"
                }}
              </h2>
              <p>
                Write <code>output/index.md</code> for a report.<br />Save plots
                to <code>output/</code> to preview them here.
              </p>
              <p class="subtle">Relative image links work automatically.</p>
            </div>
            <ImageGallery v-if="images.length" :images="images"
              ><h3>All generated images</h3></ImageGallery
            >
          </template>
          <ImageGallery v-if="tab === 'images'" :images="images"
            ><p v-if="!images.length" class="subtle">
              No generated images yet.
            </p></ImageGallery
          >
          <div v-if="tab === 'files'" class="artifact-list">
            <p class="subtle" v-if="!artifacts.length">
              Generated files will appear here.
            </p>
            <a
              v-for="artifact in artifacts"
              :key="artifact.path"
              :href="artifact.url"
              :download="artifact.path.split('/').pop()"
              ><span>{{ artifact.path }}</span
              ><span>{{ (artifact.size / 1024).toFixed(1) }} KB ↓</span></a
            >
          </div>
          <pre v-if="tab === 'logs'" class="logs">{{
            result?.logs ||
            (running
              ? "Waiting for execution to finish. Package setup is included in the run time."
              : "stdout and stderr appear here after a run.")
          }}</pre>
        </div>
        <div class="output-footer">
          output/index.md → report <span>output/* → files & images</span>
        </div>
      </section>
    </main>
    <input type="file" multiple ref="uploadInput" @change="upload" hidden />
  </div>
</template>
