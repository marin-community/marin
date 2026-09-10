import { createRouter, createWebHistory } from 'vue-router'
import Browse from './views/Browse.vue'
import Summary from './views/Summary.vue'
import Task from './views/Task.vue'

export const router = createRouter({
  history: createWebHistory('/tasktrove/'),
  routes: [
    { path: '/', component: Summary },
    { path: '/browse', component: Browse },
    { path: '/t/:id', component: Task, props: true },
  ],
  scrollBehavior(to, from, saved) {
    if (saved) return saved
    return to.path === from.path ? undefined : { top: 0 }
  },
})

export function taskPath(row: number): string {
  return `/t/${row}`
}
