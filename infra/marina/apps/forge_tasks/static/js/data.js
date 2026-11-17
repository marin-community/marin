// Data loading. The index (meta + one compact row per task) ships in the package and is fetched once; everything a
// task page adds comes from the backend, memoised per URL so a revisit or a prefetch never refetches.

const cache = new Map();

function getJson(url) {
  if (!cache.has(url)) {
    const request = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      return response.json();
    });
    request.catch(() => cache.delete(url));
    cache.set(url, request);
  }
  return cache.get(url);
}

// The package's large index files ship gzipped (Marina serves them without Content-Encoding); inflate them here, and
// accept an already-inflated body in case a proxy decoded it.
function getGzJson(url) {
  if (!cache.has(url)) {
    const request = fetch(url).then(async (response) => {
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(bytes));
      return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
    });
    request.catch(() => cache.delete(url));
    cache.set(url, request);
  }
  return cache.get(url);
}

let indexPromise = null;

// {meta, tasks, byId, caps, capById, subjects, subjectById}
export function loadIndex() {
  if (!indexPromise) {
    indexPromise = Promise.all([getGzJson("data/meta.json.gz"), getGzJson("data/tasks.json.gz")]).then(([meta, table]) => {
      // Ask the backend to load this snapshot's task records now, before the reader opens one.
      fetch(`api/warm?snap=${meta.snapshot}`).catch(() => {});
      const caps = meta.caps.map((cap, index) => ({ ...cap, index, tasks: [] }));
      const subjects = meta.subjects.map((subject) => ({ ...subject, tasks: [], caps: [] }));
      const subjectById = new Map(subjects.map((subject) => [subject.id, subject]));
      for (const cap of caps) subjectById.get(cap.subject).caps.push(cap);
      const tasks = table.rows.map((row) => {
        const task = Object.fromEntries(table.fields.map((field, i) => [field, row[i]]));
        task.capability = caps[task.cap];
        task.subject = task.capability.subject;
        task.rate = task.passes / meta.trials;
        task.revised = task.rounds > 0;
        task.search = `${task.title} ${task.id} ${task.capability.name} ${task.tags.join(" ")}`.toLowerCase();
        task.capability.tasks.push(task);
        subjectById.get(task.subject).tasks.push(task);
        return task;
      });
      return {
        meta,
        tasks,
        byId: new Map(tasks.map((task) => [task.id, task])),
        caps,
        capById: new Map(caps.map((cap) => [cap.id, cap])),
        subjects,
        subjectById,
      };
    });
  }
  return indexPromise;
}

export function subjectDetail(subject) {
  return getJson(`data/subjects/${subject}.json`);
}

export function taskDetail(meta, id) {
  return getJson(`api/detail?snap=${meta.snapshot}&id=${encodeURIComponent(id)}`);
}

export function capabilityTrials(meta, capability) {
  return getJson(`api/trials?snap=${meta.snapshot}&cap=${encodeURIComponent(capability)}`);
}

export function fullTranscript(id, trial) {
  return getJson(`api/transcript?id=${encodeURIComponent(id)}&trial=${trial}`);
}

// Warm a task page before it is opened (hover, focus, neighbours): its record and its capability's transcripts.
export function prefetchTask(meta, task) {
  taskDetail(meta, task.id).catch(() => {});
  capabilityTrials(meta, task.capability.id).catch(() => {});
}
