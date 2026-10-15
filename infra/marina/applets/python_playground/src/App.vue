<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import Workspace from "./Workspace.vue";
import { examples } from "./examples";

const record = ref(null);
const workspace = ref(null);
const error = ref("");
const loading = ref(false);
const directory = ref({ items: [], total: 0, page_size: 25 });
const query = ref("");
const sort = ref("updated");
const direction = ref("desc");
const offset = ref(0);
const deleting = ref(null);
const creating = ref(false);
const example = ref("bayesian");
const importInput = ref(null);
const legacyKey = "marina-python-playground-v1";
const browserDraft = ref(null);
try {
  browserDraft.value = localStorage.getItem(legacyKey);
} catch {
  /* The shared directory remains usable when browser storage is disabled. */
}
let activeHash = null;
let searchTimer;
let requestNumber = 0;
let routeNumber = 0;
const pageNumber = computed(
  () => Math.floor(offset.value / directory.value.page_size) + 1,
);
const pageCount = computed(() =>
  Math.max(1, Math.ceil(directory.value.total / directory.value.page_size)),
);

async function api(path, options = {}) {
  const response = await fetch(`api/${path}`, options);
  if (!response.ok) {
    const body = await response.json();
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return response.status === 204 ? null : response.json();
}
async function refresh() {
  const number = ++requestNumber;
  loading.value = true;
  try {
    const params = new URLSearchParams({
      query: query.value,
      sort: sort.value,
      direction: direction.value,
      offset: offset.value,
    });
    const value = await api(`projects?${params}`);
    if (number !== requestNumber) return;
    if (offset.value && offset.value >= value.total) {
      offset.value = 0;
      return;
    }
    directory.value = value;
  } catch (e) {
    if (number === requestNumber) error.value = e.message;
  } finally {
    if (number === requestNumber) loading.value = false;
  }
}
watch([sort, direction], () => {
  offset.value = 0;
  refresh();
});
watch(offset, refresh);
watch(query, () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    offset.value = 0;
    refresh();
  }, 250);
});
function navigate(id = "") {
  window.location.hash = id ? `/projects/${id}` : "";
}
function restoreHash() {
  history.replaceState(
    null,
    "",
    `${location.pathname}${location.search}${activeHash || ""}`,
  );
}
async function route() {
  const hash = location.hash;
  if (hash === activeHash) return;
  const number = ++routeNumber;
  if (workspace.value) {
    if (workspace.value.running) {
      error.value =
        "Wait for the current run to finish before changing projects.";
      restoreHash();
      return;
    }
    if (!(await workspace.value.saveProject())) {
      restoreHash();
      return;
    }
  }
  loading.value = true;
  error.value = "";
  try {
    const match = hash.match(/^#\/projects\/([a-f0-9-]+)$/i);
    if (hash && !match) throw new Error("Unknown project link");
    const value = match ? await api(`projects/${match[1]}`) : null;
    if (number !== routeNumber) return;
    record.value = value;
    activeHash = hash;
    deleting.value = null;
    if (!value) await refresh();
  } catch (e) {
    error.value = e.message;
    restoreHash();
  } finally {
    if (number === routeNumber) loading.value = false;
  }
}
async function createProject(document) {
  if (creating.value) return false;
  creating.value = true;
  error.value = "";
  try {
    const value = await api("projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
    });
    navigate(value.id);
    return true;
  } catch (e) {
    error.value = e.message;
    return false;
  } finally {
    creating.value = false;
  }
}
async function importProject(event) {
  try {
    const file = event.target.files[0];
    if (file) await createProject(JSON.parse(await file.text()));
  } catch (e) {
    error.value = e.message;
  }
  event.target.value = "";
}
async function importBrowserDraft() {
  try {
    if (await createProject(JSON.parse(browserDraft.value))) {
      localStorage.removeItem(legacyKey);
      browserDraft.value = null;
    }
  } catch (e) {
    error.value = e.message;
  }
}
async function deleteProject() {
  const project = deleting.value;
  try {
    await api(`projects/${project.id}?revision=${project.revision}`, {
      method: "DELETE",
    });
    deleting.value = null;
    await refresh();
  } catch (e) {
    error.value = e.message;
  }
}
function date(value) {
  return value
    ? new Date(value).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : "—";
}
function status(value) {
  return {
    never: "Not run",
    succeeded: "Completed",
    failed: "Failed",
    timed_out: "Timed out",
  }[value];
}
function sortColumn(column) {
  if (sort.value === column)
    direction.value = direction.value === "asc" ? "desc" : "asc";
  else {
    sort.value = column;
    direction.value = column === "title" ? "asc" : "desc";
  }
}
function ariaSort(column) {
  return sort.value === column
    ? direction.value === "asc"
      ? "ascending"
      : "descending"
    : "none";
}
onMounted(() => {
  window.addEventListener("hashchange", route);
  route();
});
onBeforeUnmount(() => {
  window.removeEventListener("hashchange", route);
  clearTimeout(searchTimer);
});
</script>

<template>
  <div v-if="error" class="notice error directory-error" role="alert">
    {{ error }} <button @click="error = ''">Dismiss</button>
  </div>
  <Workspace
    v-if="record"
    :key="record.id"
    :record="record"
    ref="workspace"
    @back="navigate()"
  />
  <div v-else class="app-shell">
    <header>
      <a
        class="brand"
        href="#"
        aria-label="Python playground projects"
        @click.prevent="navigate()"
      >
        <span class="brand-icon">py</span>
        <div>
          <h1>Python playground</h1>
          <p>MARINA / EXPERIMENTS</p>
        </div>
      </a>
      <div class="header-actions">
        <span class="trusted">Shared workspace · Saved in Postgres</span
        ><button @click="importInput.click()" :disabled="creating">
          Import project</button
        ><button
          class="primary"
          @click="createProject(examples.blank)"
          :disabled="creating"
        >
          + New project
        </button>
      </div>
    </header>
    <section class="project-index">
      <div class="index-heading">
        <div>
          <p class="eyebrow">YOUR WORKSPACE</p>
          <h2>Projects</h2>
          <p class="index-description">
            Python files, uploaded data and the latest report, saved together.
          </p>
        </div>
        <span class="index-count"
          >{{ directory.total }}
          {{ directory.total === 1 ? "project" : "projects" }}</span
        >
      </div>
      <div v-if="browserDraft" class="notice legacy-notice">
        A project from the earlier browser-only version is available.
        <button @click="importBrowserDraft" :disabled="creating">
          Import browser draft
        </button>
      </div>
      <div class="create-example">
        <label for="new-example">Create a new example</label>
        <select id="new-example" v-model="example" :disabled="creating">
          <option value="bayesian">Bayesian eval</option>
          <option value="seaborn">Seaborn report</option>
        </select>
        <button @click="createProject(examples[example])" :disabled="creating">
          Create example
        </button>
        <span>Creates a separate project.</span>
      </div>
      <div class="directory-controls">
        <input
          type="search"
          v-model="query"
          aria-label="Search projects"
          placeholder="Search projects…"
        />
        <div>
          <label for="project-sort">Sort by</label
          ><select id="project-sort" v-model="sort">
            <option value="updated">Last edited</option>
            <option value="title">Name</option>
            <option value="created">Created</option>
            <option value="last_run">Last run</option></select
          ><button
            @click="direction = direction === 'asc' ? 'desc' : 'asc'"
            :aria-label="
              direction === 'asc' ? 'Sort descending' : 'Sort ascending'
            "
          >
            {{ direction === "asc" ? "↑ Ascending" : "↓ Descending" }}</button
          ><button @click="refresh" :disabled="loading">Refresh</button>
        </div>
      </div>
      <div class="project-table-wrap" :aria-busy="loading">
        <table class="project-table">
          <thead>
            <tr>
              <th :aria-sort="ariaSort('title')">
                <button @click="sortColumn('title')">
                  Name
                  {{
                    sort === "title" ? (direction === "asc" ? "↑" : "↓") : ""
                  }}
                </button>
              </th>
              <th>Files</th>
              <th :aria-sort="ariaSort('updated')">
                <button @click="sortColumn('updated')">
                  Last edited
                  {{
                    sort === "updated" ? (direction === "asc" ? "↑" : "↓") : ""
                  }}
                </button>
              </th>
              <th :aria-sort="ariaSort('last_run')">
                <button @click="sortColumn('last_run')">
                  Last run
                  {{
                    sort === "last_run" ? (direction === "asc" ? "↑" : "↓") : ""
                  }}
                </button>
              </th>
              <th>Status</th>
              <th><span class="visually-hidden">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="project in directory.items" :key="project.id">
              <td>
                <a :href="`#/projects/${project.id}`" class="project-link">{{
                  project.title
                }}</a
                ><span class="project-entrypoint">{{
                  project.entrypoint || "No entrypoint"
                }}</span>
              </td>
              <td>{{ project.file_count }}</td>
              <td>{{ date(project.updated_at) }}</td>
              <td>{{ date(project.last_run_at) }}</td>
              <td>
                <span class="status-pill" :class="project.status">{{
                  status(project.status)
                }}</span>
              </td>
              <td>
                <button
                  @click="deleting = project"
                  :aria-label="`Delete ${project.title}`"
                >
                  Delete
                </button>
              </td>
            </tr>
          </tbody>
        </table>
        <div v-if="!directory.items.length" class="index-empty">
          <h3>
            {{
              loading
                ? "Loading projects…"
                : query
                  ? "No matching projects"
                  : "Start your first experiment"
            }}
          </h3>
          <p>
            {{
              query
                ? "Try a different name."
                : "Create a blank project or import one."
            }}
          </p>
        </div>
      </div>
      <div v-if="deleting" class="notice delete-notice">
        Delete <strong>{{ deleting.title }}</strong
        >, its inputs and saved outputs?
        <button @click="deleteProject">Delete project</button
        ><button @click="deleting = null">Cancel</button>
      </div>
      <div class="directory-footer">
        <span>Inputs ≤ 5 MiB · Latest outputs ≤ 10 MiB per project</span>
        <div>
          <button
            @click="offset -= directory.page_size"
            :disabled="!offset || loading"
          >
            Previous</button
          ><span>Page {{ pageNumber }} of {{ pageCount }}</span
          ><button
            @click="offset += directory.page_size"
            :disabled="
              offset + directory.page_size >= directory.total || loading
            "
          >
            Next
          </button>
        </div>
      </div>
    </section>
    <input
      type="file"
      accept=".json"
      ref="importInput"
      @change="importProject"
      hidden
    />
  </div>
</template>
