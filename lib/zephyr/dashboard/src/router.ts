import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import GraphPage from '@/pages/GraphPage.vue'
import MetricsPage from '@/pages/MetricsPage.vue'
import CountersPage from '@/pages/CountersPage.vue'
import WorkersPage from '@/pages/WorkersPage.vue'

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'graph', component: GraphPage },
  { path: '/metrics', name: 'metrics', component: MetricsPage },
  { path: '/counters', name: 'counters', component: CountersPage },
  { path: '/workers', name: 'workers', component: WorkersPage },
]

export const router = createRouter({ history: createWebHistory(), routes })
