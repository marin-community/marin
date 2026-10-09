<script setup>
import { ref, computed, onMounted, defineAsyncComponent } from "vue";
import { marked } from "marked";
import DOMPurify from "dompurify";
import ValueView from "./ValueView.vue";
const PdfView = defineAsyncComponent(() => import("./PdfView.vue"));
const example =
  "s3://marin-us-east-02a/marin/data/rl/data/rl/gretel_text_to_sql-56330d3f5621db60/2026.10.07.1/final/part-00000.parquet";
const buckets = ref([]),
  entries = ref([]),
  location = ref(""),
  address = ref(""),
  preview = ref(null);
const busy = ref(false),
  error = ref(""),
  filter = ref(""),
  search = ref(""),
  tab = ref("table");
const nextToken = ref(null),
  selected = ref(null),
  visibleColumns = ref([]),
  chooseColumns = ref(false);
const sortKey = ref("name"),
  sortDirection = ref(1),
  pageSize = ref(50),
  duration = ref(0);
let generation = 0;
const recent = ref(JSON.parse(localStorage.getItem("filer.recent") || "[]"));
const filename = computed(
  () => location.value.replace(/\/$/, "").split("/").pop() || "All buckets",
);
const compact = (value) =>
  value === null
    ? "null"
    : value === undefined
      ? "—"
      : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);
const size = (value) =>
  value == null
    ? "—"
    : value < 1024
      ? `${value} B`
      : `${(value / 1024 ** Math.floor(Math.log(value) / Math.log(1024))).toFixed(1)} ${["B", "KiB", "MiB", "GiB", "TiB"][Math.floor(Math.log(value) / Math.log(1024))]}`;
const shownEntries = computed(() =>
  entries.value
    .filter((e) => e.name.toLowerCase().includes(filter.value.toLowerCase()))
    .sort(
      (a, b) =>
        Number(b.is_dir) - Number(a.is_dir) ||
        sortDirection.value *
          (sortKey.value === "size"
            ? (a.size || 0) - (b.size || 0)
            : String(a[sortKey.value] || "").localeCompare(
                String(b[sortKey.value] || ""),
              )),
    ),
);
const rowSort = ref(""),
  rowDirection = ref(1);
const markdown = computed(() =>
  DOMPurify.sanitize(marked.parse(preview.value?.text || ""), {
    FORBID_TAGS: ["img", "iframe", "style"],
    FORBID_ATTR: ["style"],
  }),
);
function sortRows(column) {
  if (rowSort.value === column) rowDirection.value *= -1;
  else {
    rowSort.value = column;
    rowDirection.value = 1;
  }
}
const rows = computed(() =>
  (preview.value?.rows || [])
    .map((row, index) => ({ row, index }))
    .filter(
      ({ row }) =>
        !search.value ||
        compact(row).toLowerCase().includes(search.value.toLowerCase()),
    )
    .sort((a, b) =>
      !rowSort.value
        ? 0
        : rowDirection.value *
          (typeof a.row[rowSort.value] === "number" &&
          typeof b.row[rowSort.value] === "number"
            ? a.row[rowSort.value] - b.row[rowSort.value]
            : compact(a.row[rowSort.value]).localeCompare(
                compact(b.row[rowSort.value]),
              )),
    ),
);
const crumbs = computed(() => {
  const match = location.value.match(/^(s3|gs):\/\/([^/]+)(?:\/(.*))?$/);
  if (!match) return [];
  const key = (match[3] || "").replace(/\/$/, "");
  const parts = [match[2], ...(key ? key.split("/") : [])];
  return parts.map((label, i) => ({
    label: label || "(empty segment)",
    url: `${match[1]}://${parts.slice(0, i + 1).join("/")}`,
  }));
});
const parent = computed(() => {
  const parts = crumbs.value;
  return parts.length > 1 ? parts[parts.length - 2].url + "/" : "";
});
async function api(route, params = {}) {
  const response = await fetch(`api/${route}?${new URLSearchParams(params)}`);
  if (!response.ok) {
    let detail;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = `Request failed (${response.status})`;
    }
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }
  return response.json();
}
function remember(url) {
  if (!url) return;
  recent.value = [url, ...recent.value.filter((item) => item !== url)].slice(
    0,
    8,
  );
  localStorage.setItem("filer.recent", JSON.stringify(recent.value));
}
async function open(url, kind = "auto", offset = 0, projection = null) {
  const id = ++generation,
    start = performance.now();
  busy.value = true;
  error.value = "";
  selected.value = null;
  rowSort.value = "";
  try {
    let result =
      kind === "directory" || !url || url.endsWith("/")
        ? { kind: "directory" }
        : await api("inspect", {
            url,
            offset,
            limit: pageSize.value,
            ...(projection !== null
              ? { columns: JSON.stringify(projection) }
              : {}),
          });
    let listing;
    if (result.kind === "directory") listing = await api("browse", { url });
    if (id !== generation) return;
    location.value = url;
    address.value = url;
    filter.value = "";
    search.value = "";
    if (listing) {
      entries.value = listing.entries;
      nextToken.value = listing.next_token;
      preview.value = null;
    } else {
      preview.value = result;
      visibleColumns.value = result.columns || [];
      if (!offset)
        tab.value =
          result.kind === "table"
            ? "table"
            : result.kind === "json"
              ? "json"
              : "preview";
    }
    history.replaceState(null, "", `#${encodeURIComponent(url)}`);
    remember(url);
  } catch (e) {
    if (id === generation) error.value = e.message;
  } finally {
    if (id === generation) {
      busy.value = false;
      duration.value = (performance.now() - start) / 1000;
    }
  }
}
async function more() {
  busy.value = true;
  error.value = "";
  const id = ++generation;
  try {
    const result = await api("browse", {
      url: location.value,
      token: nextToken.value,
    });
    if (id === generation) {
      entries.value.push(...result.entries);
      nextToken.value = result.next_token;
    }
  } catch (e) {
    error.value = e.message;
  } finally {
    if (id === generation) busy.value = false;
  }
}
function sort(key) {
  if (sortKey.value === key) sortDirection.value *= -1;
  else {
    sortKey.value = key;
    sortDirection.value = 1;
  }
}
async function share() {
  await navigator.clipboard.writeText(window.location.href);
}
function exportPage() {
  const blob = new Blob([JSON.stringify(preview.value.rows, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${filename.value}.page.json`;
  a.click();
  URL.revokeObjectURL(url);
}
onMounted(async () => {
  try {
    buckets.value = (await api("browse")).entries;
  } catch (e) {
    error.value = e.message;
  }
  await open(decodeURIComponent(window.location.hash.slice(1)));
  window.addEventListener("hashchange", () =>
    open(decodeURIComponent(window.location.hash.slice(1))),
  );
});
</script>

<template>
  <div class="workspace">
    <aside class="sidebar">
      <a class="brand" href="#" @click.prevent="open('')"
        ><span class="brand-mark">▤</span
        ><span>Filer<small>MARINA / DATA EXPLORER</small></span></a
      >
      <button
        class="all-buckets"
        :class="{ active: !location }"
        @click="open('')"
      >
        ▦ <span>All buckets</span
        ><span class="count">{{ buckets.length }}</span>
      </button>
      <div class="section-label">OBJECT STORAGE</div>
      <div class="bucket-list">
        <button
          v-for="bucket in buckets"
          :key="bucket.url"
          :class="{
            active:
              location.startsWith(bucket.url + '/') || location === bucket.url,
          }"
          @click="open(bucket.url + '/', 'directory')"
        >
          <span class="bucket-icon">◫</span
          ><span
            >{{ bucket.name
            }}<small>{{
              bucket.url.startsWith("gs:") ? "GOOGLE CLOUD" : "S3 COMPATIBLE"
            }}</small></span
          >
        </button>
      </div>
      <div class="section-label">RECENT LOCATIONS</div>
      <button
        v-for="url in recent.slice(0, 4)"
        :key="url"
        class="recent"
        :title="url"
        @click="open(url)"
      >
        {{ url.replace(/\/$/, "").split("/").pop() }}
      </button>
      <div class="sidebar-foot">
        <span class="status-dot"></span> Private workspace
        <a href="https://marina.oa.dev/" title="Back to Marina">↗</a>
      </div>
    </aside>
    <main>
      <header class="topbar">
        <div>
          <span class="eyebrow">WORKSPACE</span
          ><span class="top-title">Object storage</span>
        </div>
        <span class="read-only">Read only</span>
      </header>
      <form class="address-bar" @submit.prevent="open(address.trim())">
        <span class="address-icon">↗</span
        ><input
          v-model="address"
          aria-label="Storage URL"
          placeholder="Paste an s3:// or gs:// URL to explore…"
          spellcheck="false"
        /><button type="submit" :disabled="busy">Open <span>↵</span></button>
      </form>
      <nav class="breadcrumbs" aria-label="Breadcrumbs">
        <button @click="open('')">Buckets</button
        ><template v-for="(crumb, i) in crumbs" :key="crumb.url"
          ><span>/</span
          ><button
            :title="crumb.url"
            @click="
              open(crumb.url + (i === crumbs.length - 1 && preview ? '' : '/'))
            "
          >
            {{ crumb.label }}
          </button></template
        >
      </nav>
      <div v-if="error" class="error" role="alert">
        <strong>Couldn’t open this location</strong>
        <p>{{ error }}</p>
        <button @click="open(address)">Retry</button>
        <button
          @click="
            open(
              address.replace(/\/$/, '').split('/').slice(0, -1).join('/') +
                '/',
              'directory',
            )
          "
        >
          Open parent folder
        </button>
      </div>
      <div class="content" :class="{ loading: busy }" :aria-busy="busy">
        <div class="page-heading">
          <div>
            <div class="eyebrow">
              {{
                preview
                  ? preview.format + " FILE"
                  : location
                    ? "FOLDER"
                    : "YOUR STORAGE"
              }}
            </div>
            <h1>{{ filename }}</h1>
            <p v-if="!preview">
              {{
                location
                  ? "Browse objects and folders at this location."
                  : "Explore your buckets, or jump straight to a file."
              }}
            </p>
            <p v-else>
              {{ size(preview.size) }} <span>·</span>
              <template v-if="preview.total != null"
                >{{ preview.total.toLocaleString() }}
                {{ preview.truncated ? "preview" : "total" }} rows
                <span>·</span></template
              >
              <template v-if="preview.schema"
                >{{ preview.schema.length }} columns <span>·</span></template
              >{{ duration.toFixed(2) }}s
            </p>
          </div>
          <div class="heading-actions">
            <button v-if="location" @click="open(parent, 'directory')">
              ↑ Parent</button
            ><button v-if="location" @click="share">Copy link</button
            ><button @click="open(location, preview ? 'auto' : 'directory')">
              ↻ Refresh
            </button>
          </div>
        </div>
        <div v-if="busy" class="progress">Reading object storage…</div>
        <template v-if="!preview">
          <div v-if="!location" class="welcome">
            <div>
              <span class="eyebrow">JUMP INTO A DATASET</span>
              <h2>Every file, a useful view.</h2>
              <p>
                Inspect rows, expand nested records, and read long prompts
                without leaving your browser.
              </p>
              <button class="primary" @click="open(example)">
                Open a Gretel SQL dataset <span>→</span>
              </button>
            </div>
            <div class="format-stack">
              <span>PARQUET <b>Table + schema</b></span
              ><span>JSON / CSV <b>Structured records</b></span
              ><span>TEXT / IMAGES <b>Quick previews</b></span>
            </div>
          </div>
          <div class="toolbar">
            <div class="tabs">
              <span class="tab selected"
                >{{ location ? "Contents" : "Buckets" }}
                <b>{{ entries.length }}</b></span
              >
            </div>
            <input
              v-model="filter"
              aria-label="Filter files"
              placeholder="Filter names in loaded listing…"
            />
          </div>
          <div class="table-wrap">
            <table class="files">
              <thead>
                <tr>
                  <th><button @click="sort('name')">Name ↕</button></th>
                  <th><button @click="sort('size')">Size ↕</button></th>
                  <th>Type</th>
                  <th><button @click="sort('mtime')">Modified ↕</button></th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="entry in shownEntries"
                  :key="entry.url"
                  @click="
                    open(
                      entry.url +
                        (entry.is_dir && !entry.url.endsWith('/') ? '/' : ''),
                      entry.is_dir ? 'directory' : 'auto',
                    )
                  "
                >
                  <td>
                    <button class="file-link">
                      <span
                        :class="entry.is_dir ? 'folder-icon' : 'file-icon'"
                        >{{ entry.is_dir ? "▱" : "▤" }}</span
                      >{{ entry.name }}
                    </button>
                  </td>
                  <td class="mono muted">{{ size(entry.size) }}</td>
                  <td class="muted">
                    {{
                      entry.is_dir
                        ? "Folder"
                        : entry.name.split(".").pop().toUpperCase()
                    }}
                  </td>
                  <td class="muted">
                    {{
                      entry.mtime ? new Date(entry.mtime).toLocaleString() : "—"
                    }}
                  </td>
                  <td class="muted">→</td>
                </tr>
              </tbody>
            </table>
            <div v-if="!shownEntries.length && !busy" class="empty">
              {{
                filter
                  ? "No matching names in this page."
                  : "This location is empty."
              }}
            </div>
          </div>
          <footer class="listing-footer">
            <span
              >{{ shownEntries.length }} shown · one folder level · metadata
              only</span
            ><button v-if="nextToken" :disabled="busy" @click="more">
              Load next listing page
            </button>
          </footer>
        </template>
        <template v-else>
          <div v-if="preview.notice" class="notice">{{ preview.notice }}</div>
          <div v-if="preview.truncated" class="notice">
            Bounded preview: this file has more data than shown. Search and
            export cover the loaded preview only.
          </div>
          <div class="toolbar">
            <div class="tabs">
              <button
                v-if="preview.kind === 'table'"
                :class="{ selected: tab === 'table' }"
                @click="tab = 'table'"
              >
                Table</button
              ><button
                v-if="preview.schema"
                :class="{ selected: tab === 'schema' }"
                @click="tab = 'schema'"
              >
                Schema</button
              ><button
                v-if="preview.kind === 'json' || preview.kind === 'table'"
                :class="{ selected: tab === 'json' }"
                @click="tab = 'json'"
              >
                JSON</button
              ><button
                v-if="/\.(md|markdown)(\.(gz|bz2|xz))?$/i.test(location)"
                :class="{ selected: tab === 'document' }"
                @click="tab = 'document'"
              >
                Document</button
              ><button
                v-if="
                  preview.text !== undefined ||
                  ['image', 'pdf', 'audio', 'video'].includes(preview.kind)
                "
                :class="{ selected: tab === 'preview' }"
                @click="tab = 'preview'"
              >
                {{ preview.kind === "binary" ? "Hex" : "Preview" }}
              </button>
            </div>
            <div class="table-tools" v-if="preview.kind === 'table'">
              <input
                v-model="search"
                aria-label="Search loaded rows"
                placeholder="Search loaded rows…"
              /><button @click="chooseColumns = !chooseColumns">
                Columns {{ visibleColumns.length }}</button
              ><button @click="exportPage">Export page</button>
            </div>
          </div>
          <div v-if="chooseColumns && preview.schema" class="column-picker">
            <label v-for="column in preview.schema" :key="column.name"
              ><input
                type="checkbox"
                :value="column.name"
                v-model="visibleColumns"
              />{{ column.name }}</label
            ><button
              :disabled="!visibleColumns.length || busy"
              @click="
                open(location, 'auto', preview.offset, visibleColumns);
                chooseColumns = false;
              "
            >
              Read selected columns
            </button>
          </div>
          <div class="data-area" :class="{ inspecting: selected }">
            <div class="data-main">
              <div v-if="tab === 'table'" class="table-wrap data-table">
                <table>
                  <thead>
                    <tr>
                      <th class="row-number">#</th>
                      <th v-for="column in visibleColumns" :key="column">
                        <button
                          @click="sortRows(column)"
                          :title="'Sort loaded page by ' + column"
                        >
                          {{ column }} ↕</button
                        ><small>{{
                          preview.schema?.find((c) => c.name === column)?.type
                        }}</small>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="item in rows"
                      :key="item.index"
                      :class="{ chosen: selected?.index === item.index }"
                      @click="selected = item"
                    >
                      <td class="row-number">
                        {{ preview.offset + item.index + 1 }}
                      </td>
                      <td
                        v-for="column in visibleColumns"
                        :key="column"
                        :class="{ null: item.row[column] == null }"
                        :title="compact(item.row[column])"
                      >
                        <div>{{ compact(item.row[column]) }}</div>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div v-if="!rows.length" class="empty">
                  No rows match this page.
                </div>
              </div>
              <div v-else-if="tab === 'schema'" class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Data type</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="column in preview.schema" :key="column.name">
                      <td class="mono">{{ column.name }}</td>
                      <td class="mono muted">{{ column.type }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <article
                v-else-if="tab === 'document'"
                class="document-view"
                v-html="markdown"
              ></article>
              <pre v-else-if="tab === 'json'" class="source-view">{{
                JSON.stringify(preview.value ?? preview.rows, null, 2)
              }}</pre>
              <div v-else-if="preview.kind === 'image'" class="image-view">
                <img
                  :src="`api/media?${new URLSearchParams({ url: location })}`"
                  :alt="filename"
                />
              </div>
              <PdfView
                v-else-if="preview.kind === 'pdf'"
                :key="location"
                :url="location"
              />
              <div
                v-else-if="preview.kind === 'audio' || preview.kind === 'video'"
                class="image-view"
              >
                <component
                  :is="preview.kind"
                  controls
                  preload="metadata"
                  :src="`api/media?${new URLSearchParams({ url: location })}`"
                />
              </div>
              <pre v-else class="source-view">{{ preview.text }}</pre>
            </div>
            <aside v-if="selected" class="inspector">
              <div class="inspector-header">
                <div>
                  <span class="eyebrow">RECORD INSPECTOR</span>
                  <h3>Row {{ preview.offset + selected.index + 1 }}</h3>
                </div>
                <button
                  @click="selected = null"
                  aria-label="Close record inspector"
                >
                  ×
                </button>
              </div>
              <div class="fields">
                <section v-for="(value, key) in selected.row" :key="key">
                  <label>{{ key }}</label
                  ><ValueView :value="value" />
                </section>
              </div>
            </aside>
          </div>
          <footer class="listing-footer">
            <span v-if="preview.kind === 'table'"
              >Rows {{ preview.total ? preview.offset + 1 : 0 }}–{{
                preview.offset + preview.rows.length
              }}
              of {{ preview.total.toLocaleString() }} · {{ rows.length }} match
              · click a row to inspect</span
            ><span v-else>Read-only preview · contents displayed as data</span>
            <div v-if="preview.kind === 'table'" class="pagination">
              <select
                v-model="pageSize"
                aria-label="Rows per page"
                @change="open(location, 'auto', 0, visibleColumns)"
              >
                <option :value="25">25 rows</option>
                <option :value="50">50 rows</option>
                <option :value="100">100 rows</option></select
              ><button
                :disabled="!preview.offset || busy"
                @click="
                  open(
                    location,
                    'auto',
                    Math.max(0, preview.offset - pageSize),
                    visibleColumns,
                  )
                "
              >
                ← Previous</button
              ><button
                :disabled="preview.next_offset == null || busy"
                @click="
                  open(location, 'auto', preview.next_offset, visibleColumns)
                "
              >
                Next →
              </button>
            </div>
          </footer>
        </template>
      </div>
    </main>
  </div>
</template>
