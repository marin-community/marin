// One route per screen. A source, a task and a search each have an address,
// so a row is an anchor and a page survives a reload.
//
// The base is `/tasktrove/`: the kernel serves this app under that prefix and
// answers any path below it with `index.html`, so a deep path reloads.

import { createRouter, createWebHistory } from 'vue-router'
import Sources from './views/Sources.vue'
import Source from './views/Source.vue'
import Task from './views/Task.vue'
import Find from './views/Find.vue'
import Sampled from './views/Sampled.vue'

export const router = createRouter({
  history: createWebHistory('/tasktrove/'),
  routes: [
    { path: '/', component: Sources },
    { path: '/sampled', component: Sampled },
    { path: '/find', component: Find },
    { path: '/s/:source', component: Source, props: true },
    { path: '/t/:row(\\d+)', component: Task, props: (route) => ({ row: Number(route.params.row) }) },
  ],
  scrollBehavior(to, from, saved) {
    if (saved) return saved
    return to.path === from.path ? undefined : { top: 0 }
  },
})

export function sourcePath(source: string): string {
  return `/s/${encodeURIComponent(source)}`
}

export function taskPath(row: number): string {
  return `/t/${row}`
}
